#!/usr/bin/env python
"""
Post-training with the Hugging Face stack (transformers + peft + trl) on the small synthetic tasks of notebooks 07-08.

    python scripts/finetune.py sft                       # LoRA SFT of SmolLM2-135M-Instruct on booking -> JSON extraction
    python scripts/finetune.py sft --full                # full fine-tuning instead of LoRA (needs more VRAM)
    python scripts/finetune.py dpo                       # LoRA SFT, then DPO + NLL (loss_type ["sigmoid", "sft"]) with a LoRA adapter
    python scripts/finetune.py dpo --nll-weight 0        # plain DPO, to see the chosen log-probs fall (likelihood displacement)
    python scripts/finetune.py grpo --reward exact       # GRPO (LoRA) on 2-digit arithmetic with a sparse, verifiable reward
    python scripts/finetune.py grpo --reward partial     # ... or with a dense reward that gives partial credit

Every subcommand evaluates on a held-out set before and after training. The task builders, the scoring and the
reward functions below are the same ones the notebooks define inline, and tests/test_finetune_hf.py tests them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import time

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Task 1: extract a restaurant booking from a sentence as strict JSON (notebooks 07 and 08)

TRAIN_NAMES = ["Ana", "Luis", "Marta", "Pablo", "Sofia", "Diego", "Elena", "Jorge", "Lucia", "Carlos", "Irene", "Tomas",
               "Nuria", "Raul", "Clara", "Hugo", "Paula", "Adrian", "Sara", "Mario", "Laura", "Ivan", "Julia", "Victor"]
TEST_NAMES = ["Beatriz", "Oscar", "Alba", "Marcos", "Noelia", "Gonzalo", "Rocio", "Felipe", "Ines", "Samuel"]
TRAIN_PLACES = ["Casa Pepe", "La Tagliatella", "Sushi Zen", "El Faro", "Burger Hub", "Taco Loco", "Le Petit Bistro",
                "Green Garden", "La Brasa", "Mar Azul", "Pizza Napoli", "The Golden Duck", "Curry House", "El Rincon"]
TEST_PLACES = ["Asador Toro", "Bella Vista", "Noodle Bar", "Casa Lola", "The Oak Room", "Kebab Palace", "Ramen Ya"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
TEMPLATES = [
    "Hi, I'd like to book a table for {people} at {place} on {day} at {time}. The name is {name}.",
    "Can you reserve {place} for {people} people this {day} at {time}? It's under {name}.",
    "{name} here. Table for {people} at {place}, {day}, {time} please.",
    "Please make a reservation at {place}: {people} guests, {day} {time}, name {name}.",
    "We are {people}. Could we get a table at {place} on {day} around {time}? Booking for {name}.",
    "Book {place} for {name}, party of {people}, {day} at {time}.",
]
SYSTEM = "Extract the booking as JSON: name, restaurant, people (int), day (lowercase), time (24h HH:MM)."
FIELDS = ["name", "restaurant", "people", "day", "time"]


def time_text(h: int, m: int, rng: random.Random) -> str:
    """The same time written the way people write it: 8pm, 8:30 pm, 20:30 or 8 in the evening."""
    h12 = h - 12 if h > 12 else h
    style = rng.randrange(4)
    if style == 0:
        return f"{h12}{':%02d' % m if m else ''}pm"
    if style == 1:
        return f"{h12}:{m:02d} pm"
    if style == 2:
        return f"{h:02d}:{m:02d}"
    return f"{h12}{':%02d' % m if m else ''} in the evening"


def make_booking(rng: random.Random, names=TRAIN_NAMES, places=TRAIN_PLACES):
    """One (sentence, target dict) pair. The target is normalized: int people, lowercase day, 24-hour time."""
    people = rng.randint(2, 10)
    h, m = rng.randint(13, 22), rng.choice([0, 0, 15, 30, 45])
    target = {"name": rng.choice(names), "restaurant": rng.choice(places), "people": people,
              "day": rng.choice(DAYS), "time": f"{h:02d}:{m:02d}"}
    people_txt = NUM_WORDS[people] if rng.random() < 0.4 else str(people)
    day_txt = target["day"].capitalize() if rng.random() < 0.7 else target["day"]
    text = rng.choice(TEMPLATES).format(people=people_txt, place=target["restaurant"], day=day_txt,
                                        time=time_text(h, m, rng), name=target["name"])
    return text, target


def booking_examples(n: int, seed: int, split: str = "train"):
    """n examples. The test split uses names and restaurants that never appear in training."""
    rng = random.Random(seed)
    names, places = (TRAIN_NAMES, TRAIN_PLACES) if split == "train" else (TEST_NAMES, TEST_PLACES)
    return [make_booking(rng, names, places) for _ in range(n)]


def to_prompt_completion(text: str, target: dict, system: str = SYSTEM) -> dict:
    """The conversational prompt-completion format that TRL's SFTTrainer and DPOTrainer understand."""
    return {"prompt": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "completion": [{"role": "assistant", "content": json.dumps(target)}]}


def parse_json(reply: str):
    """The first {...} object in a reply as a dict, or None if there is none or it does not parse."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score_booking(reply: str, target: dict) -> dict:
    """valid = the reply contains parseable JSON; exact = every field right; fields = fraction of fields right."""
    obj = parse_json(reply)
    if obj is None:
        return {"valid": 0.0, "exact": 0.0, "fields": 0.0}
    return {"valid": 1.0, "exact": float(obj == target), "fields": sum(obj.get(k) == v for k, v in target.items()) / len(target)}


def corrupt_booking(target: dict, rng: random.Random) -> dict:
    """A plausible wrong answer that differs from the right one in one field: the mistakes a small model makes."""
    bad = dict(target)
    kind = rng.choice(["time12", "time12", "people_str", "day_cap", "name"])
    if kind == "time12":                      # forgot to convert 8 pm into 20:00
        h, m = target["time"].split(":")
        bad["time"] = f"{int(h) - 12:02d}:{m}"
    elif kind == "people_str":
        bad["people"] = str(target["people"])
    elif kind == "day_cap":
        bad["day"] = target["day"].capitalize()
    else:
        bad["name"] = rng.choice([n for n in TRAIN_NAMES if n != target["name"]])
    return bad


def booking_preferences(n: int, seed: int, system: str = SYSTEM):
    """Preference pairs for DPO: (prompt, chosen = right JSON, rejected = JSON with one realistic mistake)."""
    rng = random.Random(seed)
    rows = []
    for text, target in booking_examples(n, seed):
        rows.append({"prompt": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                     "chosen": [{"role": "assistant", "content": json.dumps(target)}],
                     "rejected": [{"role": "assistant", "content": json.dumps(corrupt_booking(target, rng))}]})
    return rows


def completion_labels(tokenizer, prompt: list, completion: list):
    """
    What "completion-only loss" does, by hand: tokenize prompt + completion with the chat template and set the
    labels of every prompt token to -100 so the cross-entropy ignores them. TRL does the same thing internally.
    """
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    full_ids = tokenizer.apply_chat_template(prompt + completion, tokenize=True, return_dict=True)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "the chat template must render the prompt as a prefix of the conversation"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


# Task 2: two-digit arithmetic with a verifiable answer (GRPO in notebook 08)

MATH_SYSTEM = "Answer with the number only."


def arithmetic_problems(n: int, seed: int):
    """Rows {"prompt": chat messages, "answer": int}: additions and subtractions of two-digit numbers."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice("+-")
        rows.append({"prompt": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": f"What is {a} {op} {b}?"}],
                     "answer": a + b if op == "+" else a - b})
    return rows


def _text(completion) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}] and plain ones as strings."""
    return completion[0]["content"] if isinstance(completion, list) else completion


def extract_number(text: str):
    """The last integer in the reply: the model may reason first, as long as it ends with the answer."""
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def reward_exact(completions, answer, **kwargs) -> list[float]:
    """Sparse reward: 1 if the last number in the reply is the right answer, else 0."""
    return [1.0 if extract_number(_text(c)) == a else 0.0 for c, a in zip(completions, answer)]


def reward_partial(completions, answer, **kwargs) -> list[float]:
    """
    Dense reward: partial credit for being close (up to 0.8) plus 0.2 for replying with just a number.
    It gives a learning signal on every sample, but it also pays for answers that are close and wrong.
    """
    out = []
    for c, a in zip(completions, answer):
        text = _text(c).strip()
        pred = extract_number(text)
        if pred is None:
            out.append(0.0)
            continue
        close = max(0.0, 1.0 - abs(pred - a) / 20)
        out.append(0.8 * close + (0.2 if re.fullmatch(r"-?\d+\.?", text) else 0.0))
    return out


# Training and evaluation (needs torch + transformers + peft + trl)

def load(model_name=MODEL, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"                                   # for batched generation
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    return tok, model.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, tok, prompts, max_new_tokens=64, batch_size=16):
    """Greedy replies for a list of chat prompts."""
    import torch
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok.apply_chat_template(prompts[i:i + batch_size], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, padding=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def eval_booking(model, tok, test):
    replies = generate(model, tok, [to_prompt_completion(x, y)["prompt"] for x, y in test])
    scores = [score_booking(r, y) for r, (_, y) in zip(replies, test)]
    return {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}


def eval_math(model, tok, rows):
    replies = generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=32)
    return sum(reward_exact(replies, [r["answer"] for r in rows])) / len(rows)


def lora_config(rank=16):
    from peft import LoraConfig
    return LoraConfig(r=rank, lora_alpha=2 * rank, target_modules="all-linear", task_type="CAUSAL_LM")


def common_args(args, out, micro_batch=4, accum=4):
    import torch
    return dict(output_dir=out, per_device_train_batch_size=micro_batch, gradient_accumulation_steps=accum, logging_steps=5,
                save_strategy="no", report_to="none", bf16=torch.cuda.is_available(), gradient_checkpointing=True,
                disable_tqdm=True, seed=args.seed)


def run_sft(args, tok, model, n_train=None, epochs=None):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    train = Dataset.from_list([to_prompt_completion(x, y) for x, y in booking_examples(n_train or args.n_train, args.seed)])
    peft_config = None if args.full else lora_config(args.rank)
    cfg = SFTConfig(num_train_epochs=epochs or args.epochs, learning_rate=(args.lr if args.stage == "sft" else None) or (1e-4 if args.full else 1e-3),
                    lr_scheduler_type="cosine", warmup_steps=5, max_length=256, optim="adafactor" if args.full else "adamw_torch",
                    **common_args(args, os.path.join(args.out, "sft")))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, peft_config=peft_config)
    trainer.train()
    m = trainer.model
    return m.merge_and_unload() if peft_config is not None else m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["sft", "dpo", "grpo"])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA for SFT")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (alpha = 2 * rank)")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--nll-weight", type=float, default=1.0, help="DPO: weight of the SFT loss on the chosen answer (0 = plain DPO)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--reward", choices=["exact", "partial"], default="exact", help="GRPO reward function")
    p.add_argument("--grpo-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "llm_edu_finetune"), help="trainer scratch directory")
    args = p.parse_args()

    from transformers import set_seed
    set_seed(args.seed)
    tok, model = load(args.model)
    t0 = time.time()

    if args.stage == "sft":
        test = booking_examples(40, 1, "test")
        print("before:", eval_booking(model, tok, test))
        model = run_sft(args, tok, model)
        print("after SFT:", eval_booking(model, tok, test))

    elif args.stage == "dpo":
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
        test = booking_examples(40, 1, "test")
        model = run_sft(args, tok, model)                    # a partially trained SFT model (about 30 % exact), so DPO has room to help
        print("SFT:", eval_booking(model, tok, test))
        losses = ["sigmoid", "sft"] if args.nll_weight > 0 else ["sigmoid"]
        weights = [1.0, args.nll_weight] if args.nll_weight > 0 else None
        # A LoRA adapter on top of the SFT model: the reference model is the same network with the adapter switched off.
        cfg = DPOConfig(beta=args.beta, loss_type=losses, loss_weights=weights, learning_rate=args.lr or 2e-4, num_train_epochs=1,
                        lr_scheduler_type="cosine", warmup_steps=3, max_length=256, **common_args(args, os.path.join(args.out, "dpo")))
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(booking_preferences(400, 5)), processing_class=tok,
                             peft_config=lora_config(args.rank))
        trainer.train()
        logs = [h for h in trainer.state.log_history if "logps/chosen" in h]
        print(f"chosen log-prob {logs[0]['logps/chosen']:.2f} -> {logs[-1]['logps/chosen']:.2f} | "
              f"rejected {logs[0]['logps/rejected']:.2f} -> {logs[-1]['logps/rejected']:.2f}")
        print("after DPO" + (" + NLL" if args.nll_weight else "") + ":", eval_booking(trainer.model, tok, test))

    else:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        test = arithmetic_problems(100, 999)
        print("before: exact accuracy", eval_math(model, tok, test))
        # 32 completions per optimizer step = 4 prompts x 8 samples (8 per micro-batch x 4 accumulation steps).
        cfg = GRPOConfig(num_generations=8, max_completion_length=32, max_steps=args.grpo_steps, learning_rate=args.lr or 1e-4,
                         beta=0.04, temperature=1.0, lr_scheduler_type="constant",
                         **common_args(args, os.path.join(args.out, "grpo"), micro_batch=8, accum=4))
        reward = reward_exact if args.reward == "exact" else reward_partial
        trainer = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=Dataset.from_list(arithmetic_problems(2000, 0)),
                              processing_class=tok, peft_config=lora_config(args.rank))
        trainer.train()
        print("after GRPO: exact accuracy", eval_math(trainer.model, tok, test))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
