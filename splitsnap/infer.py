# splitsnap/infer.py
"""infer.py - generation, JSON parsing, optional per-field confidence."""
import re
import torch
from .schema import TASK_START, ensure_schema
from .preprocess import prepare_image
from .confidence import field_confidences


def parse_sequence(processor, seq):
    tok = processor.tokenizer
    seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()          # drop the leading task token
    return processor.token2json(seq)


@torch.inference_mode()
def generate_raw(model, processor, img, device="cuda", max_length=512, task=TASK_START, use_crop=False):
    """The decoded text exactly as generated (debugging: shows what the model really emits)."""
    tok = processor.tokenizer
    half = next(model.parameters()).dtype == torch.float16
    pv = processor(prepare_image(img, use_crop), return_tensors="pt").pixel_values.to(device)
    pv = pv.half() if half else pv
    dec = torch.tensor([[tok.convert_tokens_to_ids(task)]], device=device)
    out = model.generate(pv, decoder_input_ids=dec, max_length=max_length, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, use_cache=True, num_beams=1, bad_words_ids=[[tok.unk_token_id]])
    return tok.decode(out[0], skip_special_tokens=False)


@torch.inference_mode()
def generate_json(model, processor, img, device="cuda", max_length=768, with_conf=False, task=TASK_START, use_crop=False,
                  fast_prep=False, block_unk=True):
    tok = processor.tokenizer
    half = next(model.parameters()).dtype == torch.float16
    pv = processor(prepare_image(img, use_crop, fast=fast_prep), return_tensors="pt").pixel_values.to(device)
    pv = pv.half() if half else pv
    dec = torch.tensor([[tok.convert_tokens_to_ids(task)]], device=device)
    out = model.generate(pv, decoder_input_ids=dec, max_length=max_length, early_stopping=True,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True,
                         num_beams=1, bad_words_ids=([[tok.unk_token_id]] if block_unk else None), return_dict_in_generate=True,
                         output_scores=with_conf)
    parsed = parse_sequence(processor, tok.decode(out.sequences[0], skip_special_tokens=False))
    pred = ensure_schema(parsed) if task == TASK_START else parsed
    if not with_conf:
        return pred
    lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    gen = out.sequences[0, 1:1 + len(lp)]                     # drop the decoder start token
    fields = field_confidences(tok, gen.tolist(), lp.exp().float().cpu().numpy())
    return pred, fields
