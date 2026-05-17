"""Dense feed-forward network implementations."""

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """SwiGLU feed-forward network used inside a Transformer block."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)
