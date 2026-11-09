#!/usr/bin/env python
"""
Pretrain a TransformerLM on a text file (or a synthetic corpus) from the command line.

Examples
--------
# synthetic corpus, tiny model, CPU — a ~2-3 minute smoke test (or `make smoke` for a faster one)
python scripts/train.py --steps 300 --out runs/smoke

# real text (e.g. tiny shakespeare), GQA + MoE, GPU
python scripts/train.py --data data/input.txt --vocab-size 2048 --d-model 384 --n-layers 6 --n-heads 6 --n-kv-heads 2 \
    --moe --n-experts 4 --steps 3000 --batch-size 32 --block-size 256 --out runs/shakespeare

Outputs: <out>/model.pt, <out>/tokenizer.json (if text), <out>/train_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from llm_edu_suite.data import load_text_file, synthetic_structured_corpus, train_val_split
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.tokenizer import ByteBPETokenizer
from llm_edu_suite.training import TrainConfig, train_lm
from llm_edu_suite.utils import get_device, human, seed_everything
from llm_edu_suite.evaluation import perplexity


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="path to a .txt file; omit for a synthetic corpus")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--tok-train-chars", type=int, default=500_000)
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--attn", default="gqa", choices=["gqa", "mla", "swa", "longformer", "blocksparse", "linear"])
    p.add_argument("--hybrid-pattern", default=None, help='e.g. "LLLG"')
    p.add_argument("--window", type=int, default=64, help="local window for swa/longformer (must be < --block-size to matter)")
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--moe", action="store_true")
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--moe-routing", default="softmax", choices=["softmax", "auxfree"])
    # training
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.data:
        text = load_text_file(args.data)
        tok = ByteBPETokenizer().train(text[: args.tok_train_chars], vocab_size=args.vocab_size, verbose=True)
        tok.save(os.path.join(args.out, "tokenizer.json"))
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        vocab_size = tok.vocab_size
        print(f"{len(text):,} chars -> {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token)")
    else:
        ids = synthetic_structured_corpus(n_tokens=300_000, vocab=args.vocab_size)
        vocab_size = args.vocab_size
        print(f"synthetic corpus: {len(ids):,} tokens")
    train_data, val_data = train_val_split(ids)

    cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=args.block_size, d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, attn=args.attn, hybrid_pattern=args.hybrid_pattern,
                      window=args.window, qk_norm=args.qk_norm, moe=args.moe, n_experts=args.n_experts, top_k=args.top_k,
                      moe_routing=args.moe_routing)
    model = TransformerLM(cfg)
    print(f"model: {human(model.num_params())} params ({human(model.num_active_params())} active) on {device}")

    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, block_size=args.block_size, max_lr=args.lr, warmup=args.warmup,
                       schedule=args.schedule, grad_accum=args.grad_accum, eval_every=args.eval_every, compile=args.compile,
                       checkpoint_path=os.path.join(args.out, "model.pt"), seed=args.seed)
    res = train_lm(model, train_data, val_data, tcfg, device)
    model.save(os.path.join(args.out, "model.pt"))
    ppl = perplexity(model, val_data, block_size=args.block_size, n_batches=20, device=device)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "train": tcfg.__dict__, "val_losses": res.val_losses, "final_val_ppl": ppl,
                   "wall_time_s": res.wall_time}, f, indent=1)
    print(f"done: val perplexity {ppl:.2f} | {res.wall_time:.0f}s | saved to {args.out}/")


if __name__ == "__main__":
    main()
