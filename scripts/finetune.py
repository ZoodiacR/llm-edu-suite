#!/usr/bin/env python
"""
Post-training demo on the toy "reverse the sequence" task, evaluating the model after each stage.
DPO and GRPO are two independent branches that both start from the SFT model:

    base --SFT (full | LoRA)--> sft --DPO--> sft+dpo
                                   +--GRPO--> sft+grpo

python scripts/finetune.py                        # full SFT -> DPO+NLL / GRPO   (~5 min on CPU)
python scripts/finetune.py --lora                 # LoRA (+ trainable embeddings) instead of full SFT
python scripts/finetune.py --nll-weight 0         # plain DPO, to see likelihood displacement
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_edu_suite.data import ChatFormat, make_reverse_preferences, make_reverse_task, synthetic_structured_corpus, train_val_split
from llm_edu_suite.evaluation import exact_match_rate, perplexity_on_examples
from llm_edu_suite.finetune import apply_lora, greedy_answer, merge_lora, train_dpo, train_grpo, train_sft, trainable_parameters
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.training import TrainConfig, train_lm
from llm_edu_suite.utils import get_device, human, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=900, help="SFT steps (900 = partially trained, leaves room for DPO/GRPO)")
    p.add_argument("--lora", action="store_true", help="use LoRA (r=8, + trainable embeddings) instead of full fine-tuning for SFT")
    p.add_argument("--dpo-steps", type=int, default=300)
    p.add_argument("--nll-weight", type=float, default=1.0, help="DPO + NLL anchor on the chosen answer (0 = plain DPO)")
    p.add_argument("--grpo-iters", type=int, default=150)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    seed_everything(0)
    device = get_device(args.device)

    V = 512
    fmt = ChatFormat.for_vocab(V)
    data = synthetic_structured_corpus(n_tokens=100_000, vocab=V - 5)
    tr, va = train_val_split(data)
    base = TransformerLM(ModelConfig(vocab_size=V, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=128))
    train_lm(base, tr, va, TrainConfig(steps=200, batch_size=32, block_size=64, max_lr=3e-3, warmup=20, eval_every=200, log=False), device)
    print(f"base model {human(base.num_params())} params on {device}")

    sft_train, sft_test = make_reverse_task(fmt, n=8000, seed=0), make_reverse_task(fmt, n=200, seed=99)
    rng = random.Random(7)
    cases = [(u, u[::-1]) for u in ([rng.randint(1, 400) for _ in range(rng.randint(2, 5))] for _ in range(150))]

    def evaluate(model, name):
        em = exact_match_rate(lambda u: greedy_answer(model, fmt, [10], u, device=device), cases)
        print(f"{name:14s} exact match {100 * em:5.1f}%  reply ppl {perplexity_on_examples(model, sft_test, fmt, device=device):8.2f}")

    evaluate(base, "base")
    sft = copy.deepcopy(base)
    if args.lora:
        # The chat role tokens never appeared during pre-training, so their embeddings have to be trained too.
        apply_lora(sft, r=8, alpha=16, train_embeddings=True)
        train_sft(sft, sft_train, fmt, steps=args.steps, lr=2e-3, device=device, log_every=300, params=trainable_parameters(sft))
        merge_lora(sft)
        # After merging, the adapters no longer take part in the forward pass, but apply_lora left the base
        # weights frozen. Unfreeze them so the DPO and GRPO stages can update the merged model.
        for p in sft.parameters():
            p.requires_grad_(True)
    else:
        train_sft(sft, sft_train, fmt, steps=args.steps, lr=1e-3, device=device, log_every=300)
    evaluate(sft, "SFT" + (" (LoRA)" if args.lora else ""))

    if args.dpo_steps:
        dpo, _ = train_dpo(sft, make_reverse_preferences(fmt, n=3000, seed=1), fmt, steps=args.dpo_steps, lr=5e-5, device=device,
                           log_every=100, nll_weight=args.nll_weight)
        evaluate(dpo, "SFT+DPO" + ("+NLL" if args.nll_weight else ""))

    if args.grpo_iters:
        def reward_fn(user, resp):
            """A dense, verifiable reward: partial credit for each correct position, a penalty for the wrong length and a small bonus for ending with END."""
            body = resp[:-1] if resp and resp[-1] == fmt.end_id else resp
            target = user[::-1]
            correct = sum(a == b for a, b in zip(body, target)) / len(target)
            return correct - 0.1 * abs(len(body) - len(target)) + (0.2 if resp and resp[-1] == fmt.end_id else 0.0)
        g = random.Random(3)
        seed_everything(0)
        grpo, _ = train_grpo(copy.deepcopy(sft), fmt, lambda: [g.randint(1, 400) for _ in range(g.randint(2, 5))], reward_fn,
                             iters=args.grpo_iters, lr=1e-4, device=device, log_every=50)
        evaluate(grpo, "SFT+GRPO")


if __name__ == "__main__":
    main()
