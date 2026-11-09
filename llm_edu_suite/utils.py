"""Small utilities shared across the suite."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 0) -> None:
    """Make runs reproducible (as far as the hardware allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None, allow_mps: bool = False) -> str:
    """
    Pick the best available device: CUDA if present, otherwise CPU, unless the caller names one.
    Apple MPS is opt-in (allow_mps=True) because it only partly supports the complex tensors RoPE uses.
    """
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters in a module."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def human(n: float) -> str:
    """Format a number with K/M/B suffixes: 1234567 -> '1.23M'."""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


@contextmanager
def timer(label: str = ""):
    """Context manager that prints the elapsed wall-clock time."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{label} {dt * 1000:.1f} ms" if label else f"{dt * 1000:.1f} ms")


def show_mask(mask: torch.Tensor, name: str = "", max_t: int = 24) -> None:
    """Pretty-print a boolean (T, T) attention mask. A █ means the query in that row can attend to the key in that column."""
    m = mask[:max_t, :max_t]
    if name:
        print(f"\n{name}  — rows = query i, columns = key j")
    for i, row in enumerate(m):
        print(f"{i:2d} " + "".join("█" if v else "·" for v in row.tolist()))
