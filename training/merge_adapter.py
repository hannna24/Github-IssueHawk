# training/merge_adapter.py
"""Merge the LoRA adapter into the base model and save a standalone fp16 model.

Examples:
  # adapter stored in your HF repo under final-adapter/
  python merge_adapter.py --adapter hanabakeer24/issuehawk-qwen2.5-3b-lora \
      --subfolder final-adapter --out ../checkpoints/merged

  # adapter on local disk
  python merge_adapter.py --adapter ../checkpoints/lora-adapter --out ../checkpoints/merged
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help='local folder or HF repo id')
    ap.add_argument('--subfolder', default=None, help='folder inside the adapter repo, e.g. final-adapter')
    ap.add_argument('--base', default=BASE_MODEL)
    ap.add_argument('--out', default='../checkpoints/merged')
    ap.add_argument('--push-to', default=None, help='optional HF repo id to upload the merged model to')
    args = ap.parse_args()

    # Merge into an fp16 base, never the 4-bit one the adapter was trained on:
    # folding LoRA deltas into quantized weights is lossy.
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16, device_map='auto')
    kwargs = {'subfolder': args.subfolder} if args.subfolder else {}
    model = PeftModel.from_pretrained(base, args.adapter, **kwargs)
    merged = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f'merged model saved to {out.resolve()}')

    if args.push_to:
        merged.push_to_hub(args.push_to, private=True)
        AutoTokenizer.from_pretrained(args.base).push_to_hub(args.push_to, private=True)
        print(f'pushed to {args.push_to}')


if __name__ == '__main__':
    main()
