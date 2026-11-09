"""Notebook 04 - Feed-forward networks and MoE

Builds 04_ffn_moe.ipynb. Run: python tools/build_notebooks.py 04
"""
from _common import *  # noqa: F401,F403

# Notebook 04 - Feed-forward networks and MoE
nb = NB()
nb.md(r'''
# 04 · Feed-forward networks and Mixture of Experts

**What you will learn**
- The FFN is where ~2/3 of parameters (and most factual knowledge) live; SwiGLU vs GELU
- MoE: decouple parameters from compute with a router that picks top-k experts per token
- The load-balancing problem: auxiliary loss (Switch/Mixtral) vs bias-balanced routing (DeepSeek-V3)
- Fine-grained + shared experts (DeepSeekMoE)
- Evaluation: total vs active parameters, expert utilization, quality vs a dense model at equal *active* compute

**Source:** `llm_edu_suite/layers.py` (`SwiGLU`, `MoE`, `AuxFreeRouter`), `llm_edu_suite/model.py`.
''')
nb.code(SETUP)
nb.md(r'''
## SwiGLU
$$\text{FFN}(x) = W_{down}\big(\text{SiLU}(W_{gate}x) \odot W_{up}x\big)$$
A gated unit: one branch decides *how much* passes, the other *what*. Three matrices instead of two, so `d_ff ≈ 8/3·C` keeps the parameter count (Llama 3 8B: 4096 → 14336).
''')
nb.code(r'''
from llm_edu_suite.layers import SwiGLU, swiglu_hidden_dim
C = 128
class FFN_GELU(nn.Module):
    def __init__(self, C, d_ff):
        super().__init__(); self.w1 = nn.Linear(C, d_ff, bias=False); self.w2 = nn.Linear(d_ff, C, bias=False)
    def forward(self, x): return self.w2(F.gelu(self.w1(x)))

d_ff = swiglu_hidden_dim(C, multiple_of=8)
print(f"GELU FFN d_ff={4*C}: {count_params(FFN_GELU(C, 4*C)):,} params | SwiGLU d_ff={d_ff}: {count_params(SwiGLU(C, d_ff)):,} params")
print("Llama-3-8B rule:", swiglu_hidden_dim(4096, multiple_of=1024, ffn_dim_multiplier=1.3))
''')
nb.md(r'''
## MoE: router + experts

```
probs   = softmax(W_router x)                 # one score per expert
top_k   = best k experts, weights renormalized
output  = Σ_{e∈top_k} w_e · Expert_e(x)       # only k experts run
```
Mixtral 8x7B: 47B total / 13B active. DeepSeek-V3: 671B / 37B (256 experts, k=8, 1 shared).

**Load balancing.** Without it the router collapses onto a few experts. Two fixes:
- **Auxiliary loss** $\mathcal{L}_{aux} = E\sum_e f_e P_e$ (fraction of tokens × mean router prob per expert), minimal when uniform; added with coefficient ≈ 0.01.
- **Aux-free (DeepSeek-V3):** a per-expert bias added to the scores *only for selection*, nudged up for under-used experts and down for over-used ones after each batch. No distortion of the language objective.
''')
nb.code(r'''
from llm_edu_suite.layers import MoE, AuxFreeRouter

moe = MoE(C, d_ff=128, n_experts=8, top_k=2, n_shared=1, routing="softmax")
x = torch.randn(4, 32, C)
y, aux = moe(x)
# f_e counts routed *slots* per token, so sum_e f_e = top_k and a perfectly balanced router scores exactly top_k (2.0 here).
print("out", tuple(y.shape), f"| aux loss {aux.item():.3f} (perfectly uniform = top_k = {moe.k:.1f}; larger = imbalanced)")
print("slots per expert / tokens:", [round(v, 2) for v in moe.last_load.tolist()], f"(ideal top_k/E = {2/8:.2f} each)")
n_total = count_params(moe); n_expert = count_params(moe.experts[0]); n_active = n_total - 6 * n_expert
print(f"params total {n_total:,} | active per token {n_active:,} | ratio {n_total / n_active:.2f}x")
''')
nb.md(r'''
### Aux-free balancing in action
We force a badly imbalanced router (experts 0 and 1 receive almost every token) and watch the per-expert bias equalize the load — **without any gradient** on the router weights. After each batch the bias of over-used experts goes down by `bias_lr` and the bias of under-used ones goes up (`bias -= lr * sign(load - mean)`).

The update size matters: the scores are sigmoids in [0, 1], so a large `bias_lr` (e.g. 0.1) overshoots and the load *oscillates* between two bad states; DeepSeek-V3 uses a small value (0.001 at scale). We plot the imbalance ratio `max load / mean load` (1.0 = perfect).
''')
nb.code(r'''
torch.manual_seed(0)
xf = torch.randn(1024, C) + 0.5                       # the inputs all share a common direction...
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for bias_lr in (0.1, 0.01):
    router = AuxFreeRouter(C, n_experts=8, top_k=2, bias_lr=bias_lr).train()
    with torch.no_grad():
        router.weight.weight[0] += 0.05; router.weight.weight[1] += 0.03   # ...and we nudge experts 0 and 1 to point along it
    imbalance, loads = [], []
    for step in range(80):
        idx, w, _ = router(xf)
        load = F.one_hot(idx, 8).sum((0, 1)).float()
        imbalance.append((load.max() / load.mean()).item()); loads.append(load)
    ax[0].plot(imbalance, label=f"bias_lr={bias_lr}")
    print(f"bias_lr={bias_lr:<5}: step 0 load {loads[0].int().tolist()}  ->  step 79 load {loads[-1].int().tolist()}"
          f"  (imbalance {imbalance[0]:.2f}x -> {imbalance[-1]:.2f}x)")
ax[0].axhline(1, color="gray", ls="--", lw=1); ax[0].set_xlabel("router update step"); ax[0].set_ylabel("max load / mean load")
ax[0].set_title("aux-free balancing: imbalance over time"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].bar(range(8), loads[0].tolist(), alpha=.5, label="step 0"); ax[1].bar(range(8), loads[-1].tolist(), alpha=.8, label="step 79 (bias_lr=0.01)")
ax[1].axhline(1024 * 2 / 8, color="gray", ls="--", lw=1, label="uniform"); ax[1].set_xlabel("expert"); ax[1].set_ylabel("tokens routed")
ax[1].set_title("load per expert"); ax[1].legend(); plt.tight_layout(); plt.show()
''')
nb.md(r'''
## Evaluation: MoE vs dense at equal *active* compute

The fair comparison is not "same total parameters" but "same FLOPs per token". We train (a) a dense model, (b) a MoE with the same active parameters but 4× total, (c) the same MoE with aux-free routing, for the same steps, and compare validation perplexity and expert utilization. On a toy corpus the MoE advantage is small or absent — MoE pays off with large data and compute; the notebook shows **how to measure**, not that MoE always wins.
''')
nb.code(r'''
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.data import synthetic_structured_corpus, train_val_split
from llm_edu_suite.training import TrainConfig, train_lm
from llm_edu_suite.evaluation import perplexity

data = synthetic_structured_corpus(n_tokens=150_000, vocab=512, n_words=120)
train_data, val_data = train_val_split(data)
tcfg = TrainConfig(steps=300, batch_size=32, block_size=64, max_lr=3e-3, warmup=30, eval_every=300, log=False)

runs = {
    "dense":          ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=256),
    "MoE 4x top-2":   ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=128, moe=True, n_experts=4, top_k=2),
    "MoE aux-free":   ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=128, moe=True, n_experts=4, top_k=2, moe_routing="auxfree"),
}
for name, cfg in runs.items():
    seed_everything(0)
    model = TransformerLM(cfg)
    res = train_lm(model, train_data, val_data, tcfg, device)
    ppl = perplexity(model, val_data, block_size=64, n_batches=20, device=device)
    loads = [[round(v, 2) for v in b.ffn.last_load.tolist()] for b in model.blocks if isinstance(b.ffn, MoE)]
    print(f"{name:14s} total {human(model.num_params()):>7}  active {human(model.num_active_params()):>7}  val ppl {ppl:7.2f}  {res.wall_time:4.1f}s")
    if loads:
        print(f"{'':14s} expert load (last batch), layer 0: {loads[0]}")
''')
nb.md(r'''
**Takeaways**
- Watch the *expert load* row: values near `[0.5, 0.5, 0.5, 0.5]` (k/E = 2/4) mean balanced routing; a dead expert shows as `0.0`.
- MoE's wall time is higher here because the expert loop is Python; grouped GEMM kernels remove that overhead in practice.
- Use MoE when compute-bound at scale with lots of data and expert-parallel infrastructure; otherwise dense. For fine-tuning, dense is far simpler.

**Next:** `05_pretraining.ipynb` — the full training pipeline end to end.
''')
nb.save(notebook_path("04_ffn_moe.ipynb"))
print("04_ffn_moe.ipynb written")
