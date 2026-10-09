# splitsnap/dataset.py
"""dataset.py - manifest-driven dataset. Each record: {id,image,gt,task,source,split}."""
import json
import torch
from torch.utils.data import Dataset
from .preprocess import prepare_image
from .augment import augment
from .tokens import json2token, collect_added_tokens


class DonutJSONLDataset(Dataset):
    def __init__(self, manifest, split, processor, max_length=768, augment_on=False,
                 sources=None, repeat=None, use_crop=False, dynamic_pad=False):
        recs = [json.loads(l) for l in open(manifest)]
        recs = [r for r in recs if r["split"] == split and (sources is None or r["source"] in sources)]
        repeat = repeat or {}
        self.recs = [r for r in recs for _ in range(repeat.get(r["source"], 1))]   # oversample e.g. real bills
        self.processor, self.max_length = processor, max_length
        self.augment_on, self.use_crop, self.dynamic_pad = augment_on, use_crop, dynamic_pad   # dynamic_pad: only valid for batch size 1
        self.added = set(collect_added_tokens())

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = prepare_image(r["image"], self.use_crop)
        if self.augment_on:
            img = augment(img, profile=r["source"])
        pv = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)
        tok = self.processor.tokenizer
        target = json2token(r["gt"], self.added) + tok.eos_token          # NO task token in the target
        ids = tok(target, add_special_tokens=False, max_length=self.max_length, padding=(False if self.dynamic_pad else "max_length"),
                  truncation=True, return_tensors="pt").input_ids.squeeze(0)
        start = tok.convert_tokens_to_ids(r["task"])
        dec_in = torch.cat([torch.tensor([start]), ids[:-1]])              # [task, t1, t2, ...] (teacher forcing)
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return {"pixel_values": pv, "decoder_input_ids": dec_in, "labels": labels}
