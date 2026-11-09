"""
Byte-level Byte-Pair Encoding (BPE) tokenizer.

Why byte-level? Starting from the 256 possible bytes means *any* string (emoji, code, any
language) can be encoded without an <UNK> token. GPT-2, Llama 3 and most modern LLMs use this
scheme, through tiktoken or Hugging Face tokenizers (both written in Rust).

Algorithm
---------
pre-tokenize:  A regex splits the text into chunks, each either a word with its leading
               whitespace or a run of whitespace. As in GPT-2, merges never cross chunk
               boundaries. That keeps tokens linguistically sensible, and it also makes training
               and encoding fast, because the work happens once per *unique* chunk rather than
               once per character.
train:         Repeat (vocab_size - 256 - specials) times: count adjacent token pairs over all
               chunks, weighting each chunk by how often it occurs, then merge the most frequent
               pair into a new token id and record the rule (a, b) -> new_id.
encode:        Pre-tokenize, then apply the learned merges to each chunk in the order they were
               learned. Results are cached per chunk.
decode:        Concatenate the bytes of every token and decode them as UTF-8.

It is pure Python and deliberately simple: the goal is to show the mechanism, not to compete
with `tiktoken` or `tokenizers`.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# A chunk is a word with its leading whitespace, or a run of whitespace. Together the chunks
# cover every character, so joining them back gives exactly the original text.
_PRETOKENIZE = re.compile(r"\s*\S+|\s+")


class ByteBPETokenizer:
    """Train / encode / decode / save / load a byte-level BPE vocabulary."""

    def __init__(self, special_tokens: tuple[str, ...] = ("<|bos|>", "<|eos|>", "<|pad|>")):
        self.merges: dict[tuple[int, int], int] = {}            # (a, b) -> merged id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}                        # name -> id, filled in after training
        self._special_names = tuple(special_tokens)
        self._cache: dict[str, list[int]] = {}                   # chunk -> ids, so repeated chunks are encoded once

    # Training

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "ByteBPETokenizer":
        """Learn merges until the vocabulary (bytes + merges + specials) reaches vocab_size."""
        n_merges = vocab_size - 256 - len(self._special_names)
        assert n_merges > 0, "vocab_size must exceed 256 + number of special tokens"
        # Count each unique chunk once; everything below works per unique chunk, weighted by its count.
        chunk_counts = Counter(_PRETOKENIZE.findall(text))
        chunks = {c: list(c.encode("utf-8")) for c in chunk_counts}
        for i in range(n_merges):
            pairs: Counter = Counter()
            for c, ids in chunks.items():
                if len(ids) < 2:
                    continue
                w = chunk_counts[c]
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += w
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            new_id = 256 + i
            for c, ids in chunks.items():
                if len(ids) >= 2:
                    chunks[c] = self._merge(ids, (a, b), new_id)
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            if verbose and (i + 1) % 100 == 0:
                print(f"merge {i + 1}/{n_merges}: {self.vocab[new_id]!r} (freq {freq})")
        # Special tokens get the last ids so they never collide with learned merges.
        for name in self._special_names:
            idx = len(self.vocab)
            self.special[name] = idx
            self.vocab[idx] = name.encode("utf-8")
        self._cache.clear()
        return self

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `ids` by `new_id`."""
        out, i = [], 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # Encoding and decoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            # Apply the merge that was learned earliest (lowest id) first. This replays the
            # training order exactly, which is what makes BPE encoding deterministic.
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for chunk in _PRETOKENIZE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # A token boundary can fall in the middle of a multi-byte UTF-8 character, hence errors="replace".
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # Properties

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.special["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special["<|pad|>"]

    # Saving and loading

    def save(self, path: str) -> None:
        data = {
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
            "special": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(special_tokens=tuple(data["special"].keys()))
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        tok.special = {k: int(v) for k, v in data["special"].items()}
        for name, idx in tok.special.items():
            tok.vocab[idx] = name.encode("utf-8")
        return tok


def compression_ratio(tok: ByteBPETokenizer, text: str) -> float:
    """Bytes per token. Higher means more compression, and therefore shorter sequences for the model."""
    return len(text.encode("utf-8")) / max(1, len(tok.encode(text)))
