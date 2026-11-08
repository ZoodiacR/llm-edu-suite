"""
Post-training techniques, implemented from scratch. They work with any `nn.Module` language model
whose forward(input_ids) returns logits of shape (B, T, V). TransformerLM returns (logits, loss)
instead, so the small `logits_of` wrapper handles both cases.

Contents
--------
SFT                 supervised fine-tuning with loss masking (positions labelled -100 are ignored)
LoRA                LoRALinear, apply_lora, merge_lora, lora_state_dict
DPO                 sequence_logprob, dpo_loss, train_dpo
GRPO                grpo_loss and a complete loop that samples rollouts, scores them with a reward
                    function and updates the policy
Distillation        soft-label KL distillation loss
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE_INDEX, ChatFormat


def logits_of(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits from models that return either logits or (logits, loss)."""
    out = model(input_ids)
    return out[0] if isinstance(out, tuple) else out


# Supervised fine-tuning (SFT)

def sft_loss(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy computed only where labels != -100, i.e. on the assistant's reply."""
    logits = logits_of(model, inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def train_sft(model: nn.Module, examples, fmt: ChatFormat, steps: int = 300, batch_size: int = 32, lr: float = 1e-3,
              device: str = "cpu", log_every: int = 100, params=None, weight_decay: float = 0.0, seed: int = 0):
    """A minimal SFT loop. Pass `params` to train only a subset, for example just the LoRA adapters."""
    rng = random.Random(seed)
    model.to(device).train()
    params = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    losses = []
    for step in range(steps):
        inputs, labels = fmt.collate(rng.sample(examples, min(batch_size, len(examples))))
        loss = sft_loss(model, inputs.to(device), labels.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"sft step {step:4d}  loss {loss.item():.3f}")
    return losses


@torch.no_grad()
def greedy_answer(model: nn.Module, fmt: ChatFormat, system: list[int], user: list[int], max_new: int = 16,
                  device: str = "cpu") -> list[int]:
    """Render the prompt, decode greedily until END and return the assistant's tokens (without END)."""
    model.eval()
    seq = fmt.render_prompt(system, user)
    for _ in range(max_new):
        nxt = logits_of(model, torch.tensor([seq], device=device))[0, -1].argmax().item()
        if nxt == fmt.end_id:
            break
        seq.append(nxt)
    return seq[len(fmt.render_prompt(system, user)):]


# LoRA: low-rank adapters

class LoRALinear(nn.Module):
    """
    W' = W + (alpha / r) * B @ A, where W is frozen, A (r x in) starts random and B (out x r) starts at zero.
    Because B = 0 at init, the wrapped layer behaves exactly like the original. After training,
    `merge()` folds BA into W so inference costs nothing extra.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        dev, dt = base.weight.device, base.weight.dtype             # put the adapters on the same device and dtype as the base layer
        self.A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=dt))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        return self.base(x) + self.scale * (self.drop(x) @ self.A.t() @ self.B.t())

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += (self.scale * (self.B @ self.A)).to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= (self.scale * (self.B @ self.A)).to(self.base.weight.dtype)
            self.merged = False


DEFAULT_LORA_TARGETS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0, targets=DEFAULT_LORA_TARGETS,
               train_embeddings: bool = False) -> nn.Module:
    """
    Wrap every nn.Linear whose attribute name is in `targets` and freeze everything else.

    train_embeddings=True also keeps the token embedding (and the tied lm_head) trainable. You need this
    whenever fine-tuning introduces tokens the base model has never seen, such as new chat-template role
    tokens like <|im_start|>. Their embeddings are random, and LoRA alone cannot fix them. This is the
    equivalent of `peft.LoraConfig(modules_to_save=["embed_tokens", "lm_head"])`.
    """
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
    for name, p in model.named_parameters():
        is_adapter = name.endswith(".A") or name.endswith(".B")
        is_embedding = train_embeddings and ("tok_emb" in name or "lm_head" in name)
        p.requires_grad_(is_adapter or is_embedding)
    return model


def lora_parameters(model: nn.Module):
    """Only the adapter matrices A and B."""
    return [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]


def trainable_parameters(model: nn.Module):
    """Every parameter with requires_grad=True: the adapters plus anything apply_lora left trainable."""
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter weights. This is what you save and share; for a 7B model it is just a few MB."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}


def merge_lora(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    return model


# DPO: direct preference optimization

def sequence_logprob(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, average: bool = False) -> torch.Tensor:
    """log pi(y | x): the sum (or mean) of the token log-probs where labels != -100. Returns shape (B,)."""
    logits = logits_of(model, inputs)
    logp = logits.log_softmax(dim=-1)
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)
    tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    s = (tok * mask).sum(-1)
    return s / mask.sum(-1).clamp(min=1) if average else s


def dpo_loss(policy: nn.Module, ref: nn.Module, chosen, rejected, beta: float = 0.1, nll_weight: float = 0.0):
    """
    L = -log sigmoid( beta * [ (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ] )
        + nll_weight * NLL(y_w)                                             (optional, "DPO + NLL" / RPO)
    `chosen` and `rejected` are (inputs, labels) tensor pairs. Returns (loss, metrics dict).

    Why the NLL term? DPO only cares about the *margin* between chosen and rejected, and it can widen
    that margin by pushing both log-probabilities down ("likelihood displacement"), which hurts
    generation. This is very visible when chosen and rejected differ by a single token, as in our toy
    task. Adding the SFT loss on the chosen answer anchors it (Pang et al. 2024, "Iterative Reasoning
    Preference Optimization").
    """
    pi_w, pi_l = sequence_logprob(policy, *chosen), sequence_logprob(policy, *rejected)
    with torch.no_grad():
        ref_w, ref_l = sequence_logprob(ref, *chosen), sequence_logprob(ref, *rejected)
    r_w, r_l = beta * (pi_w - ref_w), beta * (pi_l - ref_l)          # DPO's implicit rewards
    loss = -F.logsigmoid(r_w - r_l).mean()
    if nll_weight:
        loss = loss + nll_weight * sft_loss(policy, *chosen)
    metrics = {"loss": loss.item(), "accuracy": (r_w > r_l).float().mean().item(),
               "margin": (r_w - r_l).mean().item(), "chosen_logp": pi_w.mean().item(), "rejected_logp": pi_l.mean().item()}
    return loss, metrics


def train_dpo(sft_model: nn.Module, pairs, fmt: ChatFormat, steps: int = 150, batch_size: int = 16, lr: float = 1e-4,
              beta: float = 0.1, device: str = "cpu", log_every: int = 50, seed: int = 0, nll_weight: float = 0.0):
    """The policy starts from the SFT model, and the reference is a frozen copy of that same model."""
    rng = random.Random(seed)
    policy = copy.deepcopy(sft_model).to(device).train()
    ref = copy.deepcopy(sft_model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    history = []
    for step in range(steps):
        batch = rng.sample(pairs, min(batch_size, len(pairs)))
        ci, cl = fmt.collate([c for c, _ in batch])
        ri, rl = fmt.collate([r for _, r in batch])
        loss, m = dpo_loss(policy, ref, (ci.to(device), cl.to(device)), (ri.to(device), rl.to(device)), beta, nll_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        m["step"] = step
        history.append(m)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"dpo step {step:4d}  loss {m['loss']:.3f}  acc {m['accuracy']:.2f}  margin {m['margin']:+.2f}")
    return policy, history


# GRPO: group relative policy optimization

def grpo_loss(policy: nn.Module, ref: nn.Module, old_logprobs: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor,
              rewards: torch.Tensor, group_size: int, beta: float = 0.04, eps: float = 0.2):
    """
    Group Relative Policy Optimization (DeepSeekMath / DeepSeek-R1).
      advantages  A_i = (r_i - mean_group) / std_group          (no critic needed)
      policy loss  -mean( min(rho * A, clip(rho, 1-eps, 1+eps) * A) ),  rho = pi / pi_old
      KL penalty   Schulman's unbiased estimator  exp(ref - pi) - (ref - pi) - 1
    `rewards` and `old_logprobs` have shape (P * G,), laid out group by group.

    `train_grpo` takes a single gradient step per sampled batch, so pi == pi_old, rho == 1 and the clip
    never kicks in. That is exactly GRPO with mu = 1 inner iterations, which amounts to REINFORCE with a
    group baseline plus a KL penalty. The ratio and clip terms start to matter when you reuse the same
    samples for several updates (mu > 1), as PPO does.
    """
    r = rewards.view(-1, group_size)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    logp = sequence_logprob(policy, inputs, labels)
    with torch.no_grad():
        ref_logp = sequence_logprob(ref, inputs, labels)
    ratio = (logp - old_logprobs).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
    d = ref_logp - logp
    kl = (d.exp() - d - 1).mean()
    return pg + beta * kl, {"pg": pg.item(), "kl": kl.item(), "reward": rewards.mean().item()}


@torch.no_grad()
def sample_group(policy: nn.Module, fmt: ChatFormat, system: list[int], user: list[int], G: int, max_new: int = 8,
                 temperature: float = 1.0, device: str = "cpu"):
    """Sample G responses for one prompt, returned as token lists (ending with END when it was produced)."""
    prompt = fmt.render_prompt(system, user)
    seqs = [list(prompt) for _ in range(G)]
    end_at = [None] * G
    for _ in range(max_new):
        logits = logits_of(policy, torch.tensor(seqs, device=device))[:, -1] / temperature
        nxt = torch.multinomial(logits.softmax(-1), 1).squeeze(-1).tolist()
        for i in range(G):
            seqs[i].append(nxt[i])                       # keep appending even after END so the batch stays rectangular
            if end_at[i] is None and nxt[i] == fmt.end_id:
                end_at[i] = len(seqs[i])
        if all(e is not None for e in end_at):
            break
    return [s[len(prompt):(e if e is not None else len(s))] for s, e in zip(seqs, end_at)]


def train_grpo(policy: nn.Module, fmt: ChatFormat, prompt_sampler, reward_fn, iters: int = 40, prompts_per_iter: int = 8,
               group_size: int = 8, lr: float = 3e-4, beta: float = 0.04, max_new: int = 8, system=(10,),
               device: str = "cpu", log_every: int = 10):
    """
    `prompt_sampler()` returns a user token list and `reward_fn(user, response_tokens)` returns a float.
    Each iteration samples G responses per prompt, scores them, and makes one PPO-style update.
    """
    policy.to(device)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    history = []
    for it in range(iters):
        policy.eval()
        examples, rewards = [], []
        for _ in range(prompts_per_iter):
            u = prompt_sampler()
            for resp in sample_group(policy, fmt, list(system), u, group_size, max_new, device=device):
                body = resp[:-1] if resp and resp[-1] == fmt.end_id else resp
                examples.append(fmt.build_example(list(system), u, body))
                rewards.append(reward_fn(u, resp))
        inputs, labels = fmt.collate(examples)
        inputs, labels = inputs.to(device), labels.to(device)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float)
        with torch.no_grad():
            old_logp = sequence_logprob(policy, inputs, labels)
        policy.train()
        loss, info = grpo_loss(policy, ref, old_logp, inputs, labels, rewards_t, group_size, beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        info["iter"] = it
        history.append(info)
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"grpo iter {it:3d}  mean reward {info['reward']:.3f}  pg {info['pg']:+.3f}  kl {info['kl']:.4f}")
    return policy, history


# Knowledge distillation

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                      tau: float = 2.0, lam: float = 0.5) -> torch.Tensor:
    """(1 - lam) * CE(hard labels) + lam * tau^2 * KL(teacher_tau || student_tau), over labelled positions only.

    The tau^2 factor keeps the soft-label gradients on the same scale as the hard-label ones (Hinton et al. 2015).
    """
    mask = labels != IGNORE_INDEX
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX)
    p_t = F.softmax(teacher_logits / tau, dim=-1)
    log_p_s = F.log_softmax(student_logits / tau, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)
    kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    return (1 - lam) * ce + lam * tau ** 2 * kl


def train_distill(student: nn.Module, teacher: nn.Module, examples, fmt: ChatFormat, steps: int = 300, batch_size: int = 32,
                  lr: float = 2e-3, tau: float = 2.0, lam: float = 0.5, device: str = "cpu", log_every: int = 100, seed: int = 0):
    rng = random.Random(seed)
    student.to(device).train()
    teacher.to(device).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        inputs, labels = fmt.collate(rng.sample(examples, min(batch_size, len(examples))))
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            t_logits = logits_of(teacher, inputs)
        loss = distillation_loss(logits_of(student, inputs), t_logits, labels, tau, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"distill step {step:4d}  loss {loss.item():.3f}")
    return losses
