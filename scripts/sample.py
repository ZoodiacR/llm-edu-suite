#!/usr/bin/env python
"""
Generate text from a checkpoint produced by scripts/train.py.

python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --top-k 50
python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --speculative runs/small_draft   # speculative decoding
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from llm_edu_suite.inference import speculative_decode
from llm_edu_suite.model import TransformerLM
from llm_edu_suite.tokenizer import ByteBPETokenizer
from llm_edu_suite.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="directory with model.pt (+ tokenizer.json)")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--speculative", default=None, help="directory of a smaller draft model trained with the same tokenizer")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model = TransformerLM.load(os.path.join(args.run, "model.pt")).to(device)
    tok_path = os.path.join(args.run, "tokenizer.json")
    tok = ByteBPETokenizer.load(tok_path) if os.path.exists(tok_path) else None

    prompt_ids = tok.encode(args.prompt) if tok else []
    ids = torch.tensor([prompt_ids or [0]], dtype=torch.long, device=device)   # with an empty prompt we start from token id 0
    t0 = time.perf_counter()
    if args.speculative:
        # Speculative mode only uses temperature (with k=4 draft tokens); top-k, top-p and the repetition penalty are ignored.
        draft = TransformerLM.load(os.path.join(args.speculative, "model.pt")).to(device)
        out, stats = speculative_decode(model, draft, ids, args.max_new_tokens, k=4, temperature=max(args.temperature, 1e-3),
                                        eos_id=tok.eos_id if tok else None)
        print(f"[speculative] acceptance {stats['acceptance_rate']:.0%}, {stats['tokens_per_round']:.2f} tokens/target pass")
    else:
        out = model.generate(ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             eos_id=tok.eos_id if tok else None, repetition_penalty=args.repetition_penalty)
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    print(f"[{n_new} tokens in {dt:.2f}s = {n_new / dt:.1f} tok/s]\n")
    print(tok.decode(out[0].tolist()) if tok else out[0].tolist())


if __name__ == "__main__":
    main()
