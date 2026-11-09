"""Notebook 08 - Preference tuning and RL with TRL: DPO and GRPO

Builds 08_preference_rl_dpo_grpo.ipynb. Run: python tools/build_notebooks.py 08
"""
from _common import *  # noqa: F401,F403

from nb07_finetuning_sft_lora import SETUP_HF, BOOKING_TASK, BOOKING_EVAL  # same setup, task and metrics as notebook 07

nb = NB()
nb.md(r'''
# 08 · Preference tuning and RL with TRL: DPO and GRPO

SFT (notebook 07) imitates reference answers. The next two post-training stages learn from **comparisons** and from **rewards**:

**DPO** (Direct Preference Optimization) learns from pairs (prompt, *chosen*, *rejected*). It needs no reward model and no RL loop: it directly raises the chosen answer's log-probability relative to the rejected one, measured against a frozen **reference** model (the SFT model):
$$\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\Big[\underbrace{\log\tfrac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{implicit reward of chosen}} - \underbrace{\log\tfrac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{implicit reward of rejected}}\Big]\Big)$$
β controls how far the policy may drift from the reference.

**GRPO** (Group Relative Policy Optimization, DeepSeekMath / DeepSeek-R1) is RL with a **verifier**: for each prompt, sample a *group* of G answers, score each with a reward function (plain code), and use the group-normalized reward $A_i = (r_i - \bar r)/\sigma_r$ as the advantage. Answers better than their siblings become more likely, worse ones less likely. No critic network; a KL term keeps the policy near the reference.

**Plan**
1. Start from a *partially* trained SFT model on notebook 07's booking task (so there is room to improve).
2. DPO with `trl.DPOTrainer`, and its failure mode: **likelihood displacement**. Then the fix: DPO + an SFT (NLL) term.
3. GRPO with `trl.GRPOTrainer` on arithmetic, with a **sparse** (exact) vs a **dense** (partial-credit) reward.
4. Held-out evaluation of everything, and an honest discussion.
''')
nb.code(SETUP_HF)
nb.md(r'''
## 1. The starting point: a partially trained SFT model

Same task, data and metrics as notebook 07: extract a restaurant booking as strict JSON, 300 labelled examples, 40 held-out requests with unseen names and restaurants.
''')
nb.code(BOOKING_TASK)
nb.code(BOOKING_EVAL)
nb.code(r'''
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from transformers import PrinterCallback

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = os.path.join(tempfile.gettempdir(), "llm_edu_08")
tok = from_pretrained(AutoTokenizer, MODEL)
RUNS = {}

def lora(r=16):
    return LoraConfig(r=r, lora_alpha=2 * r, target_modules="all-linear", task_type="CAUSAL_LM")

def common(micro_batch, accum, logging_steps=2):
    """The training arguments every trainer below shares."""
    return dict(output_dir=OUT, per_device_train_batch_size=micro_batch, gradient_accumulation_steps=accum, bf16=BF16,
                gradient_checkpointing=True, logging_steps=logging_steps, save_strategy="no", report_to="none",
                disable_tqdm=True, seed=0)

def run(trainer, name):
    """Train, record time / peak VRAM / logs, and return the merged model."""
    trainer.remove_callback(PrinterCallback)
    reset_peak(); t0 = time.time()
    trainer.train()
    RUNS[name] = {"time": time.time() - t0, "peak": peak_gb(), "logs": trainer.state.log_history}
    print(f"{name}: {trainer.state.global_step} steps, {RUNS[name]['time']:.0f} s, peak VRAM {RUNS[name]['peak']:.2f} GB")
    return trainer.model.merge_and_unload()

cfg = SFTConfig(num_train_epochs=2, learning_rate=1e-3, lr_scheduler_type="cosine", warmup_steps=5, max_length=256, **common(8, 2))
sft_model = run(SFTTrainer(model=from_pretrained(AutoModelForCausalLM, MODEL, dtype=DTYPE).to(device), args=cfg,
                           train_dataset=Dataset.from_list(train_rows), processing_class=tok, peft_config=lora()), "SFT (LoRA)")
sft_replies = evaluate(sft_model, tok, "SFT", show=3)
SFT_STATE = {k: v.detach().cpu().clone() for k, v in sft_model.state_dict().items()}   # a CPU snapshot, to restart every method from the same SFT model
''')
nb.md(r'''
As in notebook 07, the SFT model always writes valid JSON and gets names, places, party sizes and days right. Most errors are the 12-hour → 24-hour conversion (for example writing `10:00` for "10 pm"). That is a clear target for preference tuning.

## 2. Preference pairs

DPO needs (prompt, chosen, rejected). The best rejected answers are the model's **own** mistakes: they show the model exactly what it currently does wrong (this is "on-policy" preference data). So we run the SFT model on 400 new training requests and check its answers. Where it is wrong, its answer becomes *rejected* and the right one *chosen*. Where it is right, we build a rejected answer with one realistic mistake (mostly the same time error). In a real project a human, a verifier or an LLM judge decides which answer is better; here the generator knows the truth.
''')
nb.code(r'''
def corrupt(target, rng):
    """A plausible wrong answer that differs from the right one in one field."""
    bad = dict(target)
    kind = rng.choice(["time12", "time12", "people_str", "day_cap", "name"])
    if kind == "time12":
        h, m = target["time"].split(":"); bad["time"] = f"{int(h) - 12:02d}:{m}"
    elif kind == "people_str":
        bad["people"] = str(target["people"])
    elif kind == "day_cap":
        bad["day"] = target["day"].capitalize()
    else:
        bad["name"] = rng.choice([n for n in TRAIN_NAMES if n != target["name"]])
    return bad

pref_pairs = booking_examples(400, seed=5)
own = generate(sft_model, tok, [to_prompt_completion(x, y)["prompt"] for x, y in pref_pairs])
rng = random.Random(0); pref_rows = []; n_own = 0
for (x, y), reply in zip(pref_pairs, own):
    obj = parse_json(reply)
    if obj is not None and obj != y:
        rejected = obj; n_own += 1                  # the model's own mistake
    else:
        rejected = corrupt(y, rng)
    pref_rows.append({"prompt": to_prompt_completion(x, y)["prompt"],
                      "chosen": [{"role": "assistant", "content": json.dumps(y)}],
                      "rejected": [{"role": "assistant", "content": json.dumps(rejected)}]})
print(f"{len(pref_rows)} pairs, {n_own} of them with the SFT model's own mistake as the rejected answer")
print("chosen:  ", pref_rows[0]["chosen"][0]["content"]); print("rejected:", pref_rows[0]["rejected"][0]["content"])
''')
nb.md(r'''
## 3. DPO with TRL

`DPOTrainer` takes the policy, a preference dataset and a `DPOConfig`. With a `peft_config`, we train a fresh LoRA adapter on top of the SFT model, and the reference model comes for free: it is the same network with the adapter switched off. The trainer logs what we need to diagnose DPO:
- `rewards/accuracies`: how often the implicit reward prefers chosen over rejected;
- `rewards/margins`: the mean gap between the two implicit rewards;
- `logps/chosen`, `logps/rejected`: the policy's log-probability of each answer.
''')
nb.code(r'''
DPO_LR, DPO_BETA = 2e-4, 0.1

def dpo(name, loss_type, loss_weights=None):
    sft_model.load_state_dict(SFT_STATE)            # every variant starts from the same SFT model (peft adds the adapter in place
                                                    # and merge_and_unload() removes it again, so the same object is reused)
    cfg = DPOConfig(beta=DPO_BETA, loss_type=loss_type, loss_weights=loss_weights, learning_rate=DPO_LR, num_train_epochs=1,
                    lr_scheduler_type="cosine", warmup_steps=3, max_length=256, **common(4, 4))
    trainer = DPOTrainer(model=sft_model, args=cfg, train_dataset=Dataset.from_list(pref_rows),
                         processing_class=tok, peft_config=lora())
    model = run(trainer, name)
    evaluate(model, tok, name)
    del trainer; free()

dpo("DPO", loss_type=["sigmoid"])
''')
nb.md(r'''
### Likelihood displacement, and DPO + NLL

The DPO loss only depends on the **difference** between the chosen and rejected log-ratios. Nothing in it says the chosen answer must stay likely. When chosen and rejected are almost identical (here they differ in one field, often a couple of tokens), the cheapest way to widen the margin can be to push **both** down, the rejected one a little faster. Probability mass then flows to answers that are in neither set. This is documented at scale (Razin et al. 2024, *Unintentional Unalignment*; Pal et al. 2024, *Smaug / DPO-Positive*).

The standard fix is to add the ordinary SFT loss on the chosen answer: $\mathcal{L} = \mathcal{L}_{\text{DPO}} + \lambda\,\text{NLL}(y_w)$ (Pang et al. 2024, "RPO"). In older TRL versions this was `DPOConfig(rpo_alpha=1.0)`. In TRL 1.x, losses are combined explicitly: `loss_type=["sigmoid", "sft"]` with `loss_weights=[1.0, λ]`.
''')
nb.code(r'''
dpo("DPO + NLL", loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0])

def series(name, key):
    rows = [h for h in RUNS[name]["logs"] if key in h]
    return [h["step"] for h in rows], [h[key] for h in rows]

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for name, c in [("DPO", "tab:red"), ("DPO + NLL", "tab:green")]:
    ax[0].plot(*series(name, "rewards/accuracies"), c=c, label=name)
    ax[1].plot(*series(name, "logps/chosen"), c=c, label=f"{name}: chosen")
    ax[1].plot(*series(name, "logps/rejected"), c=c, ls="--", label=f"{name}: rejected")
    ax[2].plot(*series(name, "rewards/margins"), c=c, label=name)
ax[0].set_title("preference accuracy (training batches)"); ax[1].set_title("log π(answer | prompt): chosen (solid), rejected (dashed)")
ax[2].set_title("implicit reward margin")
for a in ax: a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
for name in ["DPO", "DPO + NLL"]:
    _, ch = series(name, "logps/chosen"); _, rj = series(name, "logps/rejected")
    print(f"{name:10s} log π(chosen) {ch[0]:7.2f} -> {ch[-1]:7.2f} | log π(rejected) {rj[0]:7.2f} -> {rj[-1]:7.2f}")
''')
nb.md(r'''
**What happened.** Both runs "win" by the trainer's own metrics: preference accuracy on the training batches climbs to about 0.85 and the margin grows. But the middle plot tells two different stories.
- **Plain DPO**: the log-probability of the *correct* answer falls from −1.91 to −5.63. The right JSON becomes about 40× less likely (e^3.7), while the rejected one falls even faster (−4.81 → −12.74). The probability mass goes to answers that are in neither set, and the held-out results collapse: exact match 30 % → 7.5 %, time 30 % → 10 %, and 2 of 40 replies are no longer valid JSON. This is likelihood displacement.
- **DPO + NLL**: the chosen log-probability stays where it was (−1.91 → −2.01) while the rejected one is still pushed down (−4.81 → −8.22). That is the behaviour we wanted, and nothing breaks.

The honest part: DPO + NLL moved exact match only from 30 % to 32.5 %, which is one request out of 40 and within noise. Our pairs mostly say "this time is right, that time is wrong" for individual cases, while what the model lacks is the *rule* (12-hour → 24-hour). 400 pairs and one epoch are not enough to teach it, and more SFT data was the stronger lever in notebook 07. Preference tuning earns its keep where good answers are easier to *recognize* than to *write*: tone, helpfulness, safety, choosing between plausible solutions.

## 4. GRPO: RL with a verifiable reward

DPO needs pairs that someone labelled. GRPO only needs a **checker**: for math, compare the final number; for code, run the unit tests. That is why it powers today's reasoning models.

We switch to a task where answers are easy to verify and the model has to explore: **two-digit addition and subtraction** ("What is 47 + 38?"), starting from the instruct model itself. The system prompt asks for the number only, but the verifier is lenient, as in math RL: it reads the **last** number of the reply, so the model may reason first. With `GRPOTrainer`, reward functions are plain Python: they receive the sampled `completions` plus any dataset column (here `answer`) and return one float per completion.

**Reward design is most of the work.**
- *Sparse*: 1 if the number is exactly right, else 0. Honest, but when all G samples of a prompt are wrong, every advantage is 0 and that prompt teaches nothing.
- *Dense*: partial credit for being close (up to 0.8) + 0.2 for replying with just a number. Every sample gets a signal, but now "close but wrong" also pays, and the format bonus rewards *not* reasoning: the reward is a **proxy** for what we actually want.

To compare the two fairly, both runs compute both rewards, but only one of them drives the update (`reward_weights=[1, 0]` or `[0, 1]`). So we can watch the exact-match rate even in the run that optimizes the dense reward.
''')
nb.code(r'''
MATH_SYSTEM = "Answer with the number only."

def arithmetic_problems(n, seed):
    rng = random.Random(seed); rows = []
    for _ in range(n):
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice("+-")
        rows.append({"prompt": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": f"What is {a} {op} {b}?"}],
                     "answer": a + b if op == "+" else a - b})
    return rows

def extract_number(text):
    """The last integer in the reply: the model may reason first, as long as it ends with the answer."""
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None

def _text(c):                      # TRL passes conversational completions as [{"role": "assistant", "content": ...}]
    return c[0]["content"] if isinstance(c, list) else c

def reward_exact(completions, answer, **kwargs):
    """Sparse: 1 if the last number in the reply is the right answer."""
    return [1.0 if extract_number(_text(c)) == a else 0.0 for c, a in zip(completions, answer)]

def reward_partial(completions, answer, **kwargs):
    """Dense: up to 0.8 for being close (linear, 0 at a distance of 20 or more) + 0.2 for a bare number."""
    out = []
    for c, a in zip(completions, answer):
        text = _text(c).strip(); pred = extract_number(text)
        out.append(0.0 if pred is None else 0.8 * max(0.0, 1 - abs(pred - a) / 20) + (0.2 if re.fullmatch(r"-?\d+\.?", text) else 0.0))
    return out

math_train = Dataset.from_list(arithmetic_problems(2000, seed=0))
math_test = arithmetic_problems(100, seed=999)

def eval_math(model, name):
    replies = generate(model, tok, [r["prompt"] for r in math_test], max_new_tokens=32, batch_size=50)
    ans = [r["answer"] for r in math_test]
    MATH[name] = {"exact": np.mean(reward_exact(replies, ans)), "partial": np.mean(reward_partial(replies, ans)),
                  "abs_err": np.median([abs(extract_number(r) - a) if extract_number(r) is not None else 99 for r, a in zip(replies, ans)])}
    print(f"{name:22s} exact {100*MATH[name]['exact']:5.1f}% | mean dense reward {MATH[name]['partial']:.3f} | "
          f"median |error| {MATH[name]['abs_err']:.0f}   e.g. {math_test[0]['prompt'][1]['content']} -> {replies[0]!r}")

MATH = {}
del sft_model; free()
math_base = from_pretrained(AutoModelForCausalLM, MODEL, dtype=DTYPE).to(device)
eval_math(math_base, "instruct (no RL)")
BASE_STATE = {k: v.detach().cpu().clone() for k, v in math_base.state_dict().items()}
''')
nb.md(r'''
`GRPOConfig` essentials: `num_generations` (the group size G), `max_completion_length`, `temperature` (we need diverse samples, so 1.0), `beta` (KL coefficient; TRL's default is 0, we use 0.04 as in DeepSeekMath so we can watch the KL), and the batch size, which counts **completions**: 8 per micro-batch × 4 accumulation steps = 32 completions = 4 prompts × 8 samples per optimizer step. Rollouts are generated with `model.generate`; at scale you would set `use_vllm=True`, because sampling is the bottleneck of RL.
''')
nb.code(r'''
GRPO_STEPS, GRPO_LR = 40, 1e-4

def grpo(name, weights):
    math_base.load_state_dict(BASE_STATE)
    cfg = GRPOConfig(num_generations=8, max_completion_length=32, temperature=1.0, beta=0.04, learning_rate=GRPO_LR,
                     max_steps=GRPO_STEPS, lr_scheduler_type="constant", reward_weights=weights, **common(8, 4))
    trainer = GRPOTrainer(model=math_base, reward_funcs=[reward_exact, reward_partial], args=cfg,
                          train_dataset=math_train, processing_class=tok, peft_config=lora())
    model = run(trainer, name)
    eval_math(model, name)
    del trainer; free()

grpo("GRPO sparse (exact)", weights=[1.0, 0.0])
grpo("GRPO dense (partial)", weights=[0.0, 1.0])
''')
nb.code(r'''
def smooth(v, k=3):
    v = np.asarray(v, dtype=float); return np.convolve(v, np.ones(k) / k, mode="valid") if len(v) >= k else v

keys = ["rewards/reward_exact/mean", "rewards/reward_partial/mean", "kl", "frac_reward_zero_std", "completions/mean_length"]
titles = ["sampled answers exactly right (sparse reward)", "dense (partial-credit) reward", "KL to the reference",
          "groups where all 8 samples got the same reward", "mean completion length (tokens)"]
fig, axes = plt.subplots(2, 3, figsize=(16, 6.5)); axes = axes.flatten(); axes[-1].axis("off")
for name, c in [("GRPO sparse (exact)", "tab:blue"), ("GRPO dense (partial)", "tab:orange")]:
    for a, key in zip(axes, keys):
        steps, vals = series(name, key)
        a.plot(steps[len(steps) - len(smooth(vals)):], smooth(vals), c=c, label=name)
for a, t in zip(axes, titles):
    a.set_title(t, fontsize=10); a.set_xlabel("step"); a.grid(alpha=.3)
axes[0].legend(fontsize=8); fig.suptitle("GRPO training signals (moving average of 3 logged points)"); plt.tight_layout(); plt.show()
''')
nb.md(r'''
**What happened.**
- The instruct model starts at 5 %. It explains instead of answering, often runs out of its 32 tokens, and makes subtraction slips ("96 − 20 … gives us 94").
- After only 40 GRPO steps (160 prompts, 1,280 sampled answers, no labels, about 3.5 minutes) it reaches **46 %** with the sparse reward and **42 %** with the dense one. The median error drops from 46 to 2.
- The **completion length** falls from ~27 to ~13 tokens in both runs: the policy discovered that a short `96 - 20 = …` answer gets rewarded more often than a long explanation that gets cut off. Nobody wrote that rule; it emerged from the reward. Wrong answers remain (the example is still off by 8), but now they are close.
- **Sparse vs dense, the signal**: with the exact reward, most of the time 60–80 % of the groups (less around step 18) have all 8 samples scored the same (usually all wrong), so they contribute no gradient. With the dense reward this drops to ~5–25 %, and the exact-match rate of the samples rises faster in the first 15 steps.
- **Sparse vs dense, the outcome**: on held-out problems the dense run scores slightly *higher* on its own proxy (0.566 vs 0.553) and slightly *lower* on exact match (42 % vs 46 %). That is the direction Goodhart's law predicts, but 4 problems out of 100 is within noise, so we cannot claim it from this run. What we can say: the denser signal did not buy a better final model here.
- **KL** to the reference grows to ~0.2–0.3 nats per token: the policy changed its style a lot, as expected. In longer runs a KL that keeps climbing while the held-out metric stalls is the classic sign of reward hacking.

## 5. Evaluation
''')
nb.code(r'''
print("Booking extraction (40 held-out requests):")
for n in ["SFT", "DPO", "DPO + NLL"]:
    r = RESULTS[n]
    print(f"   {n:12s} valid {100*r['valid']:5.1f}% | exact {100*r['exact']:5.1f}% | fields {100*r['fields']:5.1f}% | time {100*r['time']:5.1f}%"
          + (f" | {RUNS[n]['time']:4.0f} s, peak {RUNS[n]['peak']:.2f} GB" if n in RUNS else ""))
print("Arithmetic (100 held-out problems):")
for n, r in MATH.items():
    print(f"   {n:22s} exact {100*r['exact']:5.1f}% | dense reward {r['partial']:.3f} | median |error| {r['abs_err']:.0f}"
          + (f" | {RUNS[n]['time']:4.0f} s, peak {RUNS[n]['peak']:.2f} GB" if n in RUNS else ""))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
names = ["SFT", "DPO", "DPO + NLL"]
ax[0].bar(names, [100 * RESULTS[n]["exact"] for n in names], color=["tab:gray", "tab:red", "tab:green"])
ax[0].set_ylabel("exact match %"); ax[0].set_title("booking extraction: preference tuning"); ax[0].set_ylim(0, 100)
names = list(MATH)
ax[1].bar(names, [100 * MATH[n]["exact"] for n in names], color=["tab:gray", "tab:blue", "tab:orange"])
ax[1].set_ylabel("exact accuracy %"); ax[1].set_title("arithmetic: RL with a verifier"); ax[1].set_ylim(0, 100)
for a in ax:
    a.grid(alpha=.3, axis="y")
    for p in a.patches: a.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, f"{p.get_height():.1f}", ha="center")
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**Reading the comparison.**
- **Preference tuning** (booking task): plain DPO *hurt* the model (30 % → 7.5 %) while its training metrics looked good. DPO + NLL was safe (32.5 %) but brought no real gain on a task where the model lacks a rule rather than a preference.
- **RL with a verifier** (arithmetic): GRPO took the model from 5 % to 42–46 % in about 3.5 minutes, without a single labelled answer, just a checker and sampling. With this budget the sparse reward was as good as the dense one on the metric that matters.
- Peak VRAM was at most 1.4 GB for everything: LoRA adapters make the reference model free (adapter off = reference), which is why DPO and GRPO fit next to Ollama on a 6 GB laptop GPU.

Small test sets (40 and 100 items) mean differences of a few points are noise; the large effects (DPO's collapse, GRPO's 5 % → 40+ %) are not.

## Takeaways

- **DPO** turns (chosen, rejected) pairs into a classification loss on implicit rewards: no reward model, no sampling. The best pairs are your model's own mistakes.
- **Watch the chosen log-probability**, not just preference accuracy: plain DPO can win the comparison while making the good answer less likely. Adding the SFT loss on the chosen answer (`loss_type=["sigmoid", "sft"]` in TRL 1.x, formerly `rpo_alpha`) is a cheap, robust default.
- **GRPO** needs only prompts and a verifier. The group-relative advantage replaces the critic, and the reward function *is* the specification: a dense reward learns from every sample but optimizes a proxy (Goodhart's law), so always evaluate on the metric you care about.
- Diagnose RL with the reward curves, the KL to the reference, and the fraction of groups with zero reward variance (they carry no gradient).
- At scale: `DPOTrainer` with a learning rate around 5e-7 to 5e-6 for full fine-tuning; `GRPOTrainer(use_vllm=True, num_generations=8–16)` with reasoning-length completions. Read: DPO (Rafailov 2023), DeepSeekMath (GRPO), DeepSeek-R1, DAPO and Dr. GRPO (the loss variants TRL offers as `loss_type`).

**Next:** `09_quantization.ipynb`.
''')
if __name__ == "__main__":
    nb.save(notebook_path("08_preference_rl_dpo_grpo.ipynb"))
    print("08_preference_rl_dpo_grpo.ipynb written")
