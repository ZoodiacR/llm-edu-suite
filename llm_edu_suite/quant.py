"""
Post-training weight quantization, from scratch.

- `quantize_absmax`     symmetric round-to-nearest, with one scale per tensor, per row or per block
- `nf4_levels`, `quantize_nf4`  4-bit NormalFloat (from QLoRA), whose levels sit at Gaussian quantiles
- `QuantizedLinear`     keeps integer weights plus scales and dequantizes on the fly (as bitsandbytes does)
- `quantize_model`      swaps every nn.Linear in a model for a quantized one
- `quantization_error`  relative RMS error, the first quality check to run before any real eval

Why block-wise? A single outlier weight forces a huge scale on the whole tensor, and then all the
small weights round to zero. Giving every 32-128 weights their own scale keeps the damage inside
one block. That is why every modern format (GPTQ, AWQ, GGUF Q4_K, NF4, the FP8 block scaling in
DeepSeek-V3) is block-wise.
"""

from __future__ import annotations

from statistics import NormalDist

import torch
import torch.nn as nn
import torch.nn.functional as F


# Uniform (affine, symmetric) quantization

def quantize_absmax(x: torch.Tensor, bits: int = 8, per_row: bool = False, block: int | None = None):
    """
    Round-to-nearest with q = round(x / s) and s = max|x| / qmax.
    Returns (q as an int8 tensor, scales, dequantized x_hat).

    By default there is one scale for the whole tensor. With per_row=True each row (output
    channel) gets its own scale, so x must be 2-D. With block=k every k consecutive elements of
    the flattened tensor share a scale, which is what modern formats do.
    """
    qmax = 2 ** (bits - 1) - 1
    shape = x.shape
    if block is not None:
        assert x.numel() % block == 0, "numel must be divisible by block"
        xb = x.reshape(-1, block)
        s = xb.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = (xb / s).round().clamp(-qmax, qmax)
        return q.reshape(shape).to(torch.int8), s, (q * s).reshape(shape)
    if per_row:
        s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    else:
        s = (x.abs().amax() / qmax).clamp(min=1e-8)
    q = (x / s).round().clamp(-qmax, qmax)
    return q.to(torch.int8), s, q * s


# NormalFloat4

def nf4_levels() -> torch.Tensor:
    """
    The 16 NF4 levels: quantiles of N(0, 1) rescaled to [-1, 1], with an exact zero in the middle
    (8 negative quantiles, 0, and 7 positive ones). Trained weights are roughly Gaussian, so each
    level ends up used about equally often, which gives less error than 16 evenly spaced levels
    with the same 4 bits.
    """
    nd = NormalDist()
    neg = [nd.inv_cdf(0.5 * i / 9) for i in range(1, 9)]
    pos = [nd.inv_cdf(0.5 + 0.5 * i / 8) for i in range(1, 8)]
    lv = torch.tensor(sorted(neg + [0.0] + pos), dtype=torch.float32)
    return lv / lv.abs().max()


_NF4 = nf4_levels()


def quantize_nf4(x: torch.Tensor, block: int = 64):
    """Block-wise NF4: divide each block by its absmax and snap every value to the nearest level. Returns (idx, scales, x_hat)."""
    assert x.numel() % block == 0
    xb = x.reshape(-1, block)
    s = xb.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    xn = xb / s
    levels = _NF4.to(x.device)
    idx = (xn.unsqueeze(-1) - levels).abs().argmin(-1)                      # (n_blocks, block), values 0..15
    x_hat = (levels[idx] * s).reshape(x.shape)
    return idx.to(torch.uint8), s, x_hat


# Quantized linear layer

class QuantizedLinear(nn.Module):
    """
    Weight-only quantized Linear, in the W4A16 / W8A16 style. We keep integer weights and
    floating-point scales in memory and dequantize them to the activation dtype inside forward.
    Memory shrinks about 4x with int4 and 2x with int8, but the matmul itself still runs in
    floating point. Real kernels fuse the dequantize and the matmul into one step.
    """

    def __init__(self, linear: nn.Linear, bits: int = 4, block: int = 64, scheme: str = "absmax"):
        super().__init__()
        W = linear.weight.detach()
        self.out_features, self.in_features = W.shape
        self.bits, self.block, self.scheme = bits, block, scheme
        if scheme == "nf4":
            assert bits == 4
            q, s, _ = quantize_nf4(W, block)
        else:
            q, s, _ = quantize_absmax(W, bits, block=block)
        self.register_buffer("q", q)                                 # int8/uint8 here; real libraries pack two 4-bit values per byte
        self.register_buffer("scales", s.to(W.dtype))                # (n_blocks, 1)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())

    def dequantize(self) -> torch.Tensor:
        if self.scheme == "nf4":
            W = (_NF4.to(self.q.device)[self.q.long()] * self.scales)
        else:
            W = self.q.reshape(-1, self.block).to(self.scales.dtype) * self.scales
        return W.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)

    def memory_bytes(self) -> int:
        """Approximate storage assuming 4-bit values are packed: bits/8 bytes per weight plus an fp16 scale per block."""
        return int(self.q.numel() * self.bits / 8) + self.scales.numel() * 2


def quantize_model(model: nn.Module, bits: int = 4, block: int = 64, scheme: str = "absmax", skip=("lm_head",)) -> nn.Module:
    """Replace the model's nn.Linear layers (except those named in `skip`) with QuantizedLinear, in place."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip and child.weight.numel() % block == 0:
                setattr(module, name, QuantizedLinear(child, bits, block, scheme))
    return model


def quantization_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    """Relative RMS error ||W - W_hat|| / ||W||."""
    return ((W - W_hat).pow(2).mean().sqrt() / W.pow(2).mean().sqrt().clamp(min=1e-12)).item()


def model_memory_bytes(model: nn.Module, dtype_bytes: int = 2) -> int:
    """Memory taken by parameters and quantized buffers, counting QuantizedLinear storage correctly."""
    total, seen = 0, set()
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            total += m.memory_bytes()
            if m.bias is not None and id(m.bias) not in seen:
                seen.add(id(m.bias))
                total += m.bias.numel() * dtype_bytes
        else:
            for p in m.parameters(recurse=False):
                if id(p) not in seen:                        # tied weights should only be counted once
                    seen.add(id(p))
                    total += p.numel() * dtype_bytes
    return total
