"""
Data utilities for the from-scratch model: corpora and batching.

Pretraining data
----------------
- `synthetic_structured_corpus`  an artificial "language" with learnable structure (fast and offline)
- `template_text_corpus`         English sentences built from templates, for offline tokenizer demos
- `load_text_file`               any .txt you have lying around (tiny shakespeare, for example)
- `TokenDataset` / `get_batch`   contiguous windows of tokens as (x, y) pairs, where y is x shifted by one

Chat templates, loss masking and preference pairs for fine-tuning are handled by Hugging Face
(`tokenizer.apply_chat_template`) and TRL in notebooks 07 and 08.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch



# Pretraining corpora

def synthetic_structured_corpus(n_tokens: int = 200_000, vocab: int = 512, n_words: int = 60,
                                sep: int = 0, seed: int = 0) -> torch.Tensor:
    """
    A stream of random "words" (2-4 tokens each, drawn from a fixed lexicon) separated by `sep`.
    Once a model has learned the lexicon, its loss drops far below the ln(vocab) of random guessing.
    """
    g = torch.Generator().manual_seed(seed)
    lexicon = [torch.randint(1, vocab, (int(torch.randint(2, 5, (1,), generator=g)),), generator=g).tolist()
               for _ in range(n_words)]
    seq: list[int] = []
    while len(seq) < n_tokens:
        seq.extend(lexicon[int(torch.randint(0, n_words, (1,), generator=g))])
        seq.append(sep)
    return torch.tensor(seq[:n_tokens], dtype=torch.long)


_SUBJECTS = ["the model", "a transformer", "the router", "our tokenizer", "the optimizer", "a small network",
             "the attention layer", "the embedding table", "the KV cache", "the dataset"]
_VERBS = ["predicts", "learns", "compresses", "selects", "normalizes", "rotates", "encodes", "stores", "routes", "evaluates"]
_OBJECTS = ["the next token", "long sequences", "rare words", "the top experts", "each vector", "query and key",
            "the input text", "past keys and values", "every token", "the validation set"]
_TAILS = ["during training.", "at inference time.", "with little memory.", "in parallel.", "without a loss spike.",
          "on a single GPU.", "for long contexts.", "step by step.", "in one forward pass.", "quite efficiently."]


def template_text_corpus(n_sentences: int = 2000, seed: int = 0) -> str:
    """Deterministic English-like text for tokenizer / tiny-LM demos without downloading anything."""
    rng = random.Random(seed)
    sents = [f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TAILS)}"
             for _ in range(n_sentences)]
    return " ".join(s[0].upper() + s[1:] for s in sents)


def load_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(data: torch.Tensor, val_frac: float = 0.1):
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
              generator: torch.Generator | None = None):
    """Sample random contiguous windows. y is x shifted by one position, i.e. the next-token targets."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class TokenDataset(torch.utils.data.Dataset):
    """Non-overlapping windows over a token tensor, for use with DataLoader."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data, self.block = data, block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block

    def __getitem__(self, i):
        s = i * self.block
        return self.data[s : s + self.block], self.data[s + 1 : s + self.block + 1]
