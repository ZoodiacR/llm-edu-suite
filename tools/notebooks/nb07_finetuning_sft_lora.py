"""Notebook 07 - Fine-tuning: SFT, LoRA and distillation

Builds 07_finetuning_sft_lora.ipynb. Run: python tools/build_notebooks.py 07
"""
from _common import *  # noqa: F401,F403

# Notebook 07 - Fine-tuning: SFT, LoRA and distillation
nb = NB()
nb.md(r'''
# 07 · Fine-tuning I: SFT, LoRA and distillation

A pretrained LM continues text. Post-training turns it into something that follows instructions.

| stage | data | objective | what changes |
|---|---|---|---|
| **SFT** | (instruction → response) pairs | next-token on the *response only* | format + instruction following |
| preferences (DPO) | (prompt, chosen, rejected) | maximize preference | tone, helpfulness, safety (→ notebook 08) |
| RL with verifiable rewards (GRPO) | problems with checkable answers | reward | reasoning (→ notebook 08) |

**This notebook:** SFT with loss masking → LoRA (train 1–2 % of parameters, merge for free inference) → distillation into a smaller student → evaluation of each on the same held-out set.

**Source:** `llm_edu_suite/finetune.py`, `llm_edu_suite/data.py` (`ChatFormat`).
''')
nb.code(SETUP_FT)
nb.code(BASE_SFT_SETUP)
nb.code(EVAL_HELPERS)
nb.md(r'''
## 1. Baseline: the base model cannot do the task
''')
nb.code(r'''
evaluate(base, "base (no SFT)")
u = [7, 42, 99, 3]
print("base answers", u, "->", greedy_answer(base, fmt, [10], u, device=device), "(expected", u[::-1], ")")
''')
nb.md(r'''
## 2. SFT with loss masking

`ChatFormat.build_example` renders `[SYS] … [END] [USER] … [END] [ASSIST] reply [END]` and sets labels to `-100` on everything but the reply. `F.cross_entropy(ignore_index=-100)` then skips those positions: the model is not trained to predict the *user's* words.

Practical defaults for real models: LR 1e-5–2e-5 (full FT) or 1e-4–2e-4 (LoRA), 1–3 epochs, pack short conversations, mix in general data to avoid forgetting. Quality beats quantity (LIMA: 1k excellent examples > 50k mediocre).
''')
nb.code(r'''
from llm_edu_suite.finetune import train_sft
sft_model = copy.deepcopy(base)
losses_full = train_sft(sft_model, sft_train, fmt, steps=1500, batch_size=32, lr=2e-3, device=device, log_every=300)
evaluate(sft_model, "full SFT")
print("SFT answers", u, "->", greedy_answer(sft_model, fmt, [10], u, device=device))
''')
nb.md(r'''
## 3. LoRA: Low-Rank Adaptation

Hypothesis: the weight *change* from fine-tuning has low rank. So freeze `W` and learn
$$W' = W + \tfrac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r\times k},\; B \in \mathbb{R}^{d\times r},\; B = 0 \text{ at init}$$

- trainable parameters drop to ~1 %; no optimizer state for `W`
- at init the model is *exactly* the base (B = 0); after training `merge()` folds `BA` into `W` → zero inference overhead
- apply to **all** linear layers (q, k, v, o, gate, up, down): better than a larger `r` on q/v only
- QLoRA = the same with the frozen base in 4-bit NF4 (→ notebook 09)
''')
nb.code(r'''
from llm_edu_suite.finetune import apply_lora, lora_parameters, trainable_parameters, lora_state_dict, merge_lora, LoRALinear

lora_naive = apply_lora(copy.deepcopy(base), r=8, alpha=16)
n_total, n_train = count_params(lora_naive), count_params(lora_naive, trainable_only=True)
print(f"total {n_total:,} | trainable (LoRA) {n_train:,} = {100*n_train/n_total:.2f}%")

x = torch.randint(0, V, (1, 10), device=device)
with torch.no_grad():
    same = torch.allclose(base(x)[0], lora_naive(x)[0], atol=1e-6)
print("identical to base at init (B = 0):", same)

losses_naive = train_sft(lora_naive, sft_train, fmt, steps=800, batch_size=32, lr=2e-3, device=device, log_every=400,
                         params=lora_parameters(lora_naive))
evaluate(lora_naive, "LoRA r=8, adapters only")
''')
nb.md(r'''
### Why plain LoRA fails here — and the fix

The loss plateaus far above full SFT and exact match stays at 0 %. Look at the role tokens: `SYS, USER, ASSIST, END, PAD` (ids 507–511) **never appeared in pre-training** (the corpus uses ids 0–506). Their embeddings are still random, and with LoRA the embedding table (and the tied `lm_head`, which must *produce* `END`) is frozen. The adapters can only re-mix features of the hidden states; they cannot teach the model what `[ASSIST]` means or how to emit `[END]`.

This is the same problem you hit in practice when you fine-tune a base model with a new chat template (`<|im_start|>`, `<|eot_id|>`…). The standard fix is to also train the embeddings: `peft.LoraConfig(modules_to_save=["embed_tokens", "lm_head"])`. Here: `apply_lora(..., train_embeddings=True)`.
''')
nb.code(r'''
seed_everything(0)
lora_model = apply_lora(copy.deepcopy(base), r=8, alpha=16, train_embeddings=True)
n_train = count_params(lora_model, trainable_only=True)
print(f"trainable: adapters {sum(p.numel() for p in lora_parameters(lora_model)):,} + embeddings {lora_model.tok_emb.weight.numel():,}"
      f" = {n_train:,} ({100*n_train/n_total:.1f}% of the model)")
losses_lora = train_sft(lora_model, sft_train, fmt, steps=1500, batch_size=32, lr=2e-3, device=device, log_every=300,
                        params=trainable_parameters(lora_model))
evaluate(lora_model, "LoRA r=8 + embeddings")

adapter = lora_state_dict(lora_model)
print(f"adapter size: {sum(v.numel() for v in adapter.values()) * 2 / 1024:.1f} KB (bf16) + embeddings "
      f"{lora_model.tok_emb.weight.numel() * 2 / 1024:.0f} KB vs full model {n_total * 2 / 1024:.0f} KB")

with torch.no_grad():
    before = lora_model(x)[0]
merge_lora(lora_model)
with torch.no_grad():
    after = lora_model(x)[0]
print("merge changes nothing:", torch.allclose(before, after, atol=1e-5))
''')
nb.code(r'''
import matplotlib.pyplot as plt
def smooth(v, k=25):
    v = np.asarray(v, dtype=float); return np.convolve(v, np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(smooth(losses_full), label="full SFT (100% of params)")
ax.plot(smooth(losses_lora), label=f"LoRA r=8 + embeddings ({100*n_train/n_total:.0f}% of params)")
ax.plot(smooth(losses_naive), label="LoRA r=8, adapters only")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("SFT loss on the reply (log scale)")
ax.set_title("SFT training curves"); ax.grid(alpha=.3); ax.legend(); plt.show()
''')
nb.md(r'''
### Rank ablation
How much rank do you need? Sweep `r` (with trainable embeddings) and watch quality vs trainable parameters. In the LoRA paper, r=4–8 already matches full fine-tuning on many tasks; on a task that needs *new* behaviour the rank matters more.
''')
nb.code(r'''
rs, ems, ns = [1, 4, 16], [], []
for r in rs:
    seed_everything(0)
    m = apply_lora(copy.deepcopy(base), r=r, alpha=2 * r, train_embeddings=True)
    train_sft(m, sft_train, fmt, steps=800, batch_size=32, lr=2e-3, device=device, log_every=0, params=trainable_parameters(m))
    em = exact_match_rate(lambda uu: greedy_answer(m, fmt, [10], uu, device=device), CASES)
    ems.append(em); ns.append(sum(p.numel() for p in lora_parameters(m)))
    print(f"r={r:2d}: adapter params {ns[-1]:>7,}  exact match {100*em:5.1f}%")
fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(rs, [100*e for e in ems], "o-"); ax.set_xscale("log", base=2)
ax.set_xlabel("LoRA rank r"); ax.set_ylabel("exact match %"); ax.set_title("quality vs rank (800 steps)"); ax.grid(alpha=.3); plt.show()
''')
nb.md(r'''
## 4. Knowledge distillation

Transfer from a strong **teacher** (our full-SFT model) to a much smaller **student** (6× fewer parameters). Two flavours:

- **(a) Sequence-level / synthetic-data distillation** — the teacher *writes the answers* for prompts that have no labels, the student does SFT on them. This is how DeepSeek-R1-Distill and most small open models are made: labelled data is expensive, prompts are cheap.
- **(b) Logit distillation** — the student matches the teacher's full output distribution with temperature τ:
  $$\mathcal{L} = (1-\lambda)\,\text{CE}(y, p_s) + \lambda\,\tau^2\,\text{KL}(p_t^{\tau} \| p_s^{\tau})$$
  The soft targets carry "dark knowledge": the full probability of every token at every position, not just which one is correct — a much denser training signal than a hard label.

The realistic scenario: we only have **1,000 human-labelled** examples, but plenty of **unlabelled prompts**.
''')
nb.code(r'''
from llm_edu_suite.finetune import train_distill
student_cfg = ModelConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
seed_everything(0); student_init = TransformerLM(student_cfg)
print(f"teacher {human(sft_model.num_params())} -> student {human(student_init.num_params())}")

human_labelled = sft_train[:1000]                                   # the small labelled set we can afford
rng_p = random.Random(123)
unlabelled_prompts = [[rng_p.randint(1, 400) for _ in range(rng_p.randint(2, 5))] for _ in range(6000)]

t0 = time.time()                                                     # the teacher answers every prompt with greedy decoding
teacher_labelled = [fmt.build_example([10], u, greedy_answer(sft_model, fmt, [10], u, device=device)) for u in unlabelled_prompts]
print(f"teacher wrote {len(teacher_labelled):,} answers in {time.time() - t0:.0f}s")

# (1) Baseline: the student sees only the 1,000 human labels.
s_human = copy.deepcopy(student_init)
train_sft(s_human, human_labelled, fmt, steps=1500, lr=2e-3, device=device, log_every=0)
evaluate(s_human, "student: 1k human labels")

# (2) Sequence-level distillation: the human labels plus the answers the teacher wrote.
s_seq = copy.deepcopy(student_init)
train_sft(s_seq, human_labelled + teacher_labelled, fmt, steps=1500, lr=2e-3, device=device, log_every=0)
evaluate(s_seq, "student: + teacher answers")

# (3) Logit distillation on the same data, mixing the hard label with the teacher's soft distribution.
s_logit = copy.deepcopy(student_init)
train_distill(s_logit, sft_model, human_labelled + teacher_labelled, fmt, steps=1500, lr=2e-3, tau=2.0, lam=0.5, device=device, log_every=0)
evaluate(s_logit, "student: + logit distill")
''')
nb.md(r'''
**What happened.** With only the 1,000 human labels the small student barely learns the task. Adding the teacher's answers to 6,000 unlabelled prompts multiplies its training data for free and transforms the result. Logit distillation (matching the teacher's full distribution with τ = 2, λ = 0.5) is the more fragile of the two: across our runs on different hardware it landed both above and below sequence-level distillation, because its benefit depends on tuning τ, λ and the learning rate. That robustness gap is one reason why sequence-level distillation (the teacher writes data, the student does plain SFT) is the default way small LLMs are distilled today. Two caveats you should always keep in mind: the student inherits the teacher's mistakes (a distilled model is bounded by its teacher), and logit distillation needs access to the teacher's logits and the **same tokenizer** — with an API-only teacher you can only do sequence-level distillation.
''')
nb.code(r'''
plot_results("Notebook 07 — every method on the same 150 held-out prompts")
''')
nb.md(r'''
## 5. Evaluation summary and what to measure in practice

| metric | what it tells you |
|---|---|
| exact match / task accuracy | did it learn the task? (your own eval set — build it *before* fine-tuning) |
| reply perplexity | calibration on held-out replies; compare checkpoints with the same tokenizer |
| general benchmarks before/after (MMLU, IFEval…) | did we break general abilities? (catastrophic forgetting) |
| LLM-judge on open-ended prompts | tone/helpfulness regressions |

Real-world equivalents: `trl.SFTTrainer(assistant_only_loss=True, packing=True)`, `peft.LoraConfig(target_modules="all-linear")`, `lm-eval-harness` for before/after benchmarks.

**Next:** `08_preference_rl_dpo_grpo.ipynb`.
''')
nb.save(notebook_path("07_finetuning_sft_lora.ipynb"))
print("07_finetuning_sft_lora.ipynb written")
