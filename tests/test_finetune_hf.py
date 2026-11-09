"""
Fast tests for the Hugging Face post-training code in scripts/finetune.py (the tasks of notebooks 07-08).

They run on CPU in about 15 seconds. The tokenizer and training tests need SmolLM2-135M-Instruct in the local
Hugging Face cache and are skipped otherwise (they never download anything).
"""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import finetune as ft  # noqa: E402


def test_booking_examples_are_deterministic_and_normalized():
    a, b = ft.booking_examples(50, 0), ft.booking_examples(50, 0)
    assert a == b
    for text, target in a:
        assert set(target) == set(ft.FIELDS)
        assert isinstance(target["people"], int) and 2 <= target["people"] <= 10
        assert target["day"] in ft.DAYS
        h, m = map(int, target["time"].split(":"))
        assert 13 <= h <= 22 and m in (0, 15, 30, 45)
        assert target["name"] in text and target["restaurant"] in text


def test_test_split_uses_unseen_names_and_restaurants():
    train = ft.booking_examples(300, 0)
    test = ft.booking_examples(40, 1, "test")
    assert not {t["name"] for _, t in test} & {t["name"] for _, t in train}
    assert not {t["restaurant"] for _, t in test} & {t["restaurant"] for _, t in train}


def test_prompt_completion_format():
    text, target = ft.booking_examples(1, 0)[0]
    ex = ft.to_prompt_completion(text, target)
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["completion"][0]["role"] == "assistant"
    assert json.loads(ex["completion"][0]["content"]) == target


def test_parse_and_score():
    target = {"name": "Ana", "restaurant": "El Faro", "people": 4, "day": "friday", "time": "20:00"}
    assert ft.score_booking(json.dumps(target), target) == {"valid": 1.0, "exact": 1.0, "fields": 1.0}
    assert ft.score_booking("Sure! " + json.dumps(target) + " Enjoy.", target)["exact"] == 1.0
    wrong = dict(target, time="08:00")
    assert ft.score_booking(json.dumps(wrong), target) == {"valid": 1.0, "exact": 0.0, "fields": 0.8}
    assert ft.score_booking('{"name": "Ana", ', target)["valid"] == 0.0
    assert ft.parse_json("[1, 2]") is None


def test_corruption_changes_exactly_one_field():
    rng = random.Random(0)
    for _, target in ft.booking_examples(200, 3):
        bad = ft.corrupt_booking(target, rng)
        assert sum(bad[k] != target[k] for k in ft.FIELDS) == 1


def test_preference_pairs():
    rows = ft.booking_preferences(20, 5)
    assert len(rows) == 20
    for r in rows:
        assert set(r) == {"prompt", "chosen", "rejected"}
        assert r["chosen"][0]["content"] != r["rejected"][0]["content"]


def test_arithmetic_and_rewards():
    rows = ft.arithmetic_problems(30, 0)
    assert rows == ft.arithmetic_problems(30, 0)
    ans = [r["answer"] for r in rows[:4]]
    good = [[{"role": "assistant", "content": str(a)}] for a in ans]
    off_by_one = [[{"role": "assistant", "content": str(a + 1)}] for a in ans]
    chatty = [[{"role": "assistant", "content": f"The answer is {a}."}] for a in ans]
    assert ft.reward_exact(good, ans) == [1.0] * 4
    assert ft.reward_exact(off_by_one, ans) == [0.0] * 4
    assert ft.reward_exact(chatty, ans) == [1.0] * 4                    # the verifier only checks the final number
    assert ft.reward_partial(good, ans) == pytest.approx([1.0] * 4)
    assert all(0.7 < r < 1.0 for r in ft.reward_partial(off_by_one, ans))  # dense reward pays for "close but wrong"
    assert all(r == pytest.approx(0.8) for r in ft.reward_partial(chatty, ans))
    assert ft.reward_partial(["no idea"], [5]) == [0.0]                   # plain-string completions work too
    assert ft.extract_number("-12 apples") == -12
    assert ft.extract_number("47 + 38 = 85") == 85                      # the last number is the answer


def _cached_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(ft.MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{ft.MODEL} is not in the local Hugging Face cache")


def test_completion_labels_mask_the_prompt():
    tok = _cached_tokenizer()
    text, target = ft.booking_examples(1, 0)[0]
    ex = ft.to_prompt_completion(text, target)
    ids, labels = ft.completion_labels(tok, ex["prompt"], ex["completion"])
    assert len(ids) == len(labels)
    n_prompt = labels.index(next(l for l in labels if l != -100))
    assert all(l == -100 for l in labels[:n_prompt]) and all(l != -100 for l in labels[n_prompt:])
    learned = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    assert learned.startswith(json.dumps(target))                    # the loss covers the JSON reply ...
    assert "<|im_end|>" in learned                                   # ... and the end-of-turn token, so it learns to stop
    assert text not in learned                                       # ... and never the user's words


def test_tiny_lora_sft_runs(tmp_path):
    """Two LoRA SFT steps with TRL on CPU (a few seconds): checks that the trainer, the dataset format and peft fit together."""
    torch = pytest.importorskip("torch")
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    tok = _cached_tokenizer()
    model = __import__("transformers").AutoModelForCausalLM.from_pretrained(ft.MODEL, local_files_only=True, dtype=torch.float32)
    train = Dataset.from_list([ft.to_prompt_completion(x, y) for x, y in ft.booking_examples(8, 0)])
    cfg = SFTConfig(output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=2, report_to="none",
                    save_strategy="no", bf16=False, use_cpu=True, disable_tqdm=True, gradient_checkpointing=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok,
                         peft_config=LoraConfig(r=4, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    assert trainer.state.global_step == 2
