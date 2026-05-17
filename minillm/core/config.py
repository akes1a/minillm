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
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
