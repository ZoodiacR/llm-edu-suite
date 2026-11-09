"""
Data utilities: corpora, batching, and SFT/preference formatting.

Pretraining data
----------------
- `synthetic_structured_corpus`  an artificial "language" with learnable structure (fast and offline)
- `template_text_corpus`         English sentences built from templates, for offline tokenizer demos
- `load_text_file`               any .txt you have lying around (tiny shakespeare, for example)
- `TokenDataset` / `get_batch`   contiguous windows of tokens as (x, y) pairs, where y is x shifted by one

SFT data
--------
`ChatFormat` renders conversations with role tokens and produces (input_ids, labels) with loss
masking: the labels are -100 everywhere except on the assistant's reply, so only the reply is
learned. `collate` pads a batch.

Preference data
---------------
`make_reverse_preferences` returns a list of (chosen, rejected) tuples. Each one is an
(input_ids, labels) pair rendered with ChatFormat for the same prompt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


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


# Chat formatting for SFT and preference data

@dataclass
class ChatFormat:
    """
    Role tokens for a minimal chat template. With a real tokenizer these would be special tokens such
    as <|im_start|>user ... <|im_end|>; here they are just reserved ids in a small vocabulary.

        [SYS] system... [END] [USER] user... [END] [ASSIST] reply... [END]
    """
    sys_id: int
    user_id: int
    assist_id: int
    end_id: int
    pad_id: int

    @classmethod
    def for_vocab(cls, vocab_size: int) -> "ChatFormat":
        """Reserve the last 5 ids of the vocabulary as role tokens, so they never collide with real text."""
        return cls(vocab_size - 5, vocab_size - 4, vocab_size - 3, vocab_size - 2, vocab_size - 1)

    def render_prompt(self, system: list[int], user: list[int]) -> list[int]:
        return [self.sys_id] + system + [self.end_id, self.user_id] + user + [self.end_id, self.assist_id]

    def build_example(self, system: list[int], user: list[int], assistant: list[int]):
        """
        Return (input_ids, labels) as tensors. labels[t] is the token that input_ids[t] should
        predict; it is -100 everywhere except on the assistant reply and its END token.
        """
        prompt = self.render_prompt(system, user)
        seq = prompt + assistant + [self.end_id]
        labels = [IGNORE_INDEX] * len(prompt) + assistant + [self.end_id]
        # Position t predicts seq[t+1], so the inputs drop the last token and the labels drop the first.
        return torch.tensor(seq[:-1]), torch.tensor(labels[1:])

    def collate(self, examples):
        """Right-pad a list of (input_ids, labels). Padding positions get label -100 so they cost nothing."""
        T = max(len(i) for i, _ in examples)
        inputs = torch.full((len(examples), T), self.pad_id, dtype=torch.long)
        labels = torch.full((len(examples), T), IGNORE_INDEX, dtype=torch.long)
        for r, (i, l) in enumerate(examples):
            inputs[r, : len(i)] = i
            labels[r, : len(l)] = l
        return inputs, labels


def make_reverse_task(fmt: ChatFormat, n: int = 512, min_len: int = 2, max_len: int = 5,
                      token_range=(1, 400), system=(10,), seed: int = 0):
    """
    A toy instruction task: the assistant must return the user's token sequence reversed.
    Solving it means copying and reordering tokens through attention, a real (if tiny) skill to learn by SFT.
    """
    rng = random.Random(seed)
    lo, hi = token_range[0], min(token_range[1], fmt.sys_id - 1)      # stay inside the vocab and below the role tokens
    data = []
    for _ in range(n):
        u = [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]
        data.append(fmt.build_example(list(system), u, u[::-1]))
    return data


def make_reverse_preferences(fmt: ChatFormat, n: int = 512, system=(10,), seed: int = 0,
                             token_range=(1, 400), min_len: int = 2, max_len: int = 5):
    """Preference pairs. The chosen answer is the correct reversal; the rejected one is correct except
    for its last token, a deliberately subtle error."""
    rng = random.Random(seed)
    lo, hi = token_range[0], min(token_range[1], fmt.sys_id - 1)      # stay inside the vocab and below the role tokens
    data = []
    for _ in range(n):
        u = [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]
        good = u[::-1]
        bad = good[:-1] + [rng.choice([t for t in range(lo, hi + 1) if t != good[-1]])]
        data.append((fmt.build_example(list(system), u, good), fmt.build_example(list(system), u, bad)))
    return data
