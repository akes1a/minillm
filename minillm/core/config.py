"""Model configuration objects."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    hidden_size: int
    ffn_hidden_size: int
    dropout: float = 0.0
    use_moe: bool = False

