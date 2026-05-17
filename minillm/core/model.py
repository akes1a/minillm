"""Decoder-only language model entry point."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from minillm.core.attention import CausalSelfAttention
from minillm.core.config import ModelConfig
from minillm.core.ffn import FeedForward
from minillm.core.norm import RMSNorm


@dataclass
class MiniLLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with causal attention and dense FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config.hidden_size, config.n_heads, config.dropout)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = FeedForward(config.hidden_size, config.ffn_hidden_size, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """Minimal decoder-only language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> MiniLLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )

        return MiniLLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            input_window = input_ids[:, -self.config.max_seq_len :]
            logits = self(input_window).logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    @staticmethod
    def estimate_parameters(config: ModelConfig) -> int:
        """Estimate parameter count without allocating model weights."""
        config.validate()

        embeddings = config.vocab_size * config.hidden_size
        position_embeddings = config.max_seq_len * config.hidden_size

        attention = 4 * config.hidden_size * config.hidden_size
        ffn = 3 * config.hidden_size * config.ffn_hidden_size
        norms = 2 * config.hidden_size
        per_layer = attention + ffn + norms

        final_norm = config.hidden_size
        lm_head = 0 if config.tie_embeddings else config.hidden_size * config.vocab_size

        return (
            embeddings
            + position_embeddings
            + config.n_layers * per_layer
            + final_norm
            + lm_head
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
