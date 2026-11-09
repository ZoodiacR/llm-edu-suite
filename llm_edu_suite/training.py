"""
Training utilities: LR schedules, optimizer groups, a complete training loop, checkpoints.

This is the same loop used to pretrain real LLMs; what changes at scale is the data loader and
the multi-GPU distribution (DDP / FSDP). It includes:
  - a warmup followed by a cosine (or WSD) learning-rate schedule
  - AdamW with weight decay on the matrices only, not on norm gains or biases
  - gradient accumulation, to simulate batches larger than memory allows
  - gradient clipping, logging the pre-clip grad norm because it is a key health signal
  - mixed precision: bf16 autocast on CUDA, or fp16 with a GradScaler when bf16 is unsupported
  - periodic validation and checkpointing
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .data import get_batch


# Learning-rate schedules

def lr_schedule(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1, kind: str = "cosine") -> float:
    """
    kind="cosine": a linear warmup, then a cosine decay down to min_ratio * max_lr (GPT-3, Llama).
    kind="wsd":    warmup, a long constant ("stable") phase, then a linear decay over the last 20%
                   of training (MiniCPM, DeepSeek). Because the LR stays flat until the end, you
                   can extend training without deciding the total length up front.
    """
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if kind == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
    if kind == "wsd":
        decay_start = int(0.8 * total)
        if step < decay_start:
            return max_lr
        p = (step - decay_start) / max(1, total - decay_start)
        return max_lr * (1 - (1 - min_ratio) * min(1.0, p))
    raise ValueError(kind)


# Optimizer

def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95), fused: bool | None = None):
    """
    AdamW with two parameter groups: matrices (dim >= 2) get weight decay, while vectors such as
    norm gains and biases do not. We follow the LLM convention of beta2 = 0.95 rather than 0.999,
    which makes training more robust to loss spikes.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {}
    if fused is None:
        fused = torch.cuda.is_available() and any(p.is_cuda for p in decay)
    if fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)


# Training loop

@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 32
    block_size: int = 64
    max_lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 50
    schedule: str = "cosine"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 1
    eval_every: int = 100
    eval_batches: int = 10
    amp: bool | None = None            # None means automatic: on for CUDA, off elsewhere
    compile: bool = False
    log: bool = True
    checkpoint_path: str | None = None
    seed: int = 0


@dataclass
class TrainResult:
    train_losses: list = field(default_factory=list)   # (step, loss)
    val_losses: list = field(default_factory=list)     # (step, loss)
    grad_norms: list = field(default_factory=list)
    lrs: list = field(default_factory=list)
    wall_time: float = 0.0


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, block_size: int, batch_size: int, device: str, n_batches: int = 10) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_lm(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, cfg: TrainConfig, device: str = "cpu") -> TrainResult:
    """Pretrain `model` on token tensors with next-token prediction and return the loss curves."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    opt = make_optimizer(model, cfg.max_lr, cfg.weight_decay)
    use_amp = (device == "cuda") if cfg.amp is None else cfg.amp
    # Use bf16 everywhere except on CUDA GPUs that lack bf16 support, where we fall back to fp16 with a
    # GradScaler. bf16 never needs a scaler, so CPU runs don't get one.
    amp_dtype = torch.float16 if (device == "cuda" and not torch.cuda.is_bf16_supported()) else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16 and device == "cuda"))
    step_model = torch.compile(model) if cfg.compile else model

    res = TrainResult()
    t0 = time.time()
    for step in range(cfg.steps):
        lr = lr_schedule(step, cfg.max_lr, cfg.warmup, cfg.steps, cfg.min_lr_ratio, cfg.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
            with torch.autocast(device_type=device if device != "mps" else "cpu", dtype=amp_dtype, enabled=use_amp):
                _, loss = step_model(x, y)
            scaler.scale(loss / cfg.grad_accum).backward()
            loss_acc += loss.item() / cfg.grad_accum

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        res.train_losses.append((step, loss_acc))
        res.grad_norms.append(float(gnorm))
        res.lrs.append(lr)

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            vl = estimate_loss(model, val_data, cfg.block_size, cfg.batch_size, device, cfg.eval_batches)
            res.val_losses.append((step, vl))
            if cfg.log:
                print(f"step {step:5d} | train {loss_acc:.3f} | val {vl:.3f} | ppl {math.exp(vl):7.1f} "
                      f"| lr {lr:.2e} | grad {float(gnorm):.2f} | {time.time() - t0:5.1f}s")
            if cfg.checkpoint_path:
                torch.save({"config": getattr(model, "cfg", None) and model.cfg.to_dict(),
                            "state_dict": model.state_dict(), "step": step}, cfg.checkpoint_path)
    res.wall_time = time.time() - t0
    return res


def plot_training(res: TrainResult, title: str = "training"):
    """Plot loss, learning rate and grad norm. Matplotlib is imported here so the module still works without it."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    s, l = zip(*res.train_losses)
    ax[0].plot(s, l, label="train", alpha=0.7)
    if res.val_losses:
        vs, vl = zip(*res.val_losses)
        ax[0].plot(vs, vl, "o-", label="val")
    ax[0].set_title(f"{title}: loss"); ax[0].set_xlabel("step"); ax[0].legend()
    ax[1].plot(res.lrs); ax[1].set_title("learning rate"); ax[1].set_xlabel("step")
    ax[2].plot(res.grad_norms); ax[2].set_title("grad norm (pre-clip)"); ax[2].set_xlabel("step"); ax[2].set_yscale("log")
    plt.tight_layout()
    return fig
