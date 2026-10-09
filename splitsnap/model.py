# splitsnap/model.py
"""model.py - build Donut-base for our schema (new tokens, new input size) / reload a checkpoint."""
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor
from .tokens import collect_added_tokens
from .schema import TASK_START

BASE = "naver-clova-ix/donut-base"


def build_model(height=1280, width=960, max_length=768, base=BASE):
    config = VisionEncoderDecoderConfig.from_pretrained(base)
    config.encoder.image_size = [height, width]            # (height, width); keep multiples of 32
    config.decoder.max_length = max_length
    processor = DonutProcessor.from_pretrained(base)
    processor.image_processor.size = {"height": height, "width": width}
    processor.image_processor.do_align_long_axis = False
    # New encoder resolution => some encoder weights are freshly initialised; training still works
    # (the official CORD config says so too).
    model = VisionEncoderDecoderModel.from_pretrained(base, config=config)
    tok = processor.tokenizer
    tok.add_tokens(collect_added_tokens(), special_tokens=True)
    model.decoder.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(TASK_START)
    return model, processor


def load_model(path):
    return VisionEncoderDecoderModel.from_pretrained(path), DonutProcessor.from_pretrained(path)


def sanity_check(model, processor, height, width):
    """Run once after build_model: catches the classic (height,width) vs (width,height) mix-up."""
    from PIL import Image
    pv = processor(Image.new("RGB", (500, 900), "white"), return_tensors="pt").pixel_values
    assert tuple(pv.shape[-2:]) == (height, width), f"processor gives {tuple(pv.shape[-2:])}, expected {(height, width)}"
    assert model.decoder.get_input_embeddings().num_embeddings >= len(processor.tokenizer)
    print("sanity OK:", tuple(pv.shape), "vocab", len(processor.tokenizer))
