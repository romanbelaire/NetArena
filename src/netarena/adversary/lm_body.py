"""LM body encoders: hash bag-of-features (CPU bridge) and optional HF transformers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn


class HashLmBody(nn.Module):
    """
    Deterministic bag-of-character-ngram encoder.
    Stand-in LM body for CPU smoke / bridge prototyping without weight downloads.
    """

    def __init__(self, embed_dim: int = 128, ngram: int = 3, seed: int = 0):
        super().__init__()
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        self.embed_dim = embed_dim
        self.ngram = ngram
        self.seed = seed
        # Learnable projection so the "body" can adapt slightly while staying hash-based.
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _hash_vec(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.embed_dim, dtype=torch.float32)
        padded = f"^{text.lower()}$"
        for i in range(max(0, len(padded) - self.ngram + 1)):
            gram = padded[i : i + self.ngram]
            digest = hashlib.blake2b(
                f"{self.seed}:{gram}".encode(), digest_size=8
            ).digest()
            idx = int.from_bytes(digest[:4], "little") % self.embed_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            v[idx] += sign
        norm = torch.linalg.vector_norm(v)
        if norm > 0:
            v = v / norm
        return v

    def forward(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must be non-empty")
        mat = torch.stack([self._hash_vec(t) for t in texts], dim=0)
        return self.proj(mat)


class HFLmBody(nn.Module):
    """Frozen (or partially tuned) HuggingFace encoder; mean-pool last hidden state."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 128,
        trainable: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embed_dim = int(self.encoder.config.hidden_size)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.trainable = trainable
        self.to(self.device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must be non-empty")
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        if self.trainable:
            out = self.encoder(**toks)
        else:
            with torch.no_grad():
                out = self.encoder(**toks)
        hidden = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if not self.trainable:
            pooled = pooled.detach()
        return pooled


def build_lm_body(
    backend: str,
    *,
    embed_dim: int = 128,
    model_name: str | None = None,
    seed: int = 0,
    device: str = "cpu",
    trainable_lm: bool = False,
) -> nn.Module:
    if backend == "hash":
        return HashLmBody(embed_dim=embed_dim, seed=seed)
    if backend == "hf":
        if not model_name:
            raise ValueError("model_name is required for backend='hf'")
        return HFLmBody(
            model_name,
            trainable=trainable_lm,
            device=device,
        )
    raise ValueError(f"Unknown lm backend {backend!r}; expected 'hash' or 'hf'")


def save_body_meta(path: Path, backend: str, **meta) -> None:
    import json

    path.write_text(json.dumps({"backend": backend, **meta}, indent=2))
