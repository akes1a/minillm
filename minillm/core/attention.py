"""Self-attention implementations."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        y = self.o_proj(y)
        return self.resid_dropout(y)

    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
