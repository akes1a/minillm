import torch

from minillm.core.config import ModelConfig
from minillm.core.model import MiniLLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        n_layers=2,
        n_heads=4,
        hidden_size=32,
        ffn_hidden_size=64,
        dropout=0.0,
    )


def test_forward_shapes_and_loss() -> None:
    model = MiniLLM(tiny_config())
    input_ids = torch.randint(0, 32, (2, 8))

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 8, 32)
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_generate_appends_tokens() -> None:
    model = MiniLLM(tiny_config())
    input_ids = torch.randint(0, 32, (2, 4))

    output_ids = model.generate(input_ids, max_new_tokens=3, temperature=0.0)

    assert output_ids.shape == (2, 7)


def test_target_config_parameter_count_is_expected_scale() -> None:
    config = ModelConfig(
        vocab_size=32000,
        max_seq_len=2048,
        n_layers=10,
        n_heads=20,
        hidden_size=2560,
        ffn_hidden_size=6912,
        dropout=0.0,
    )

    assert 850_000_000 < MiniLLM.estimate_parameters(config) < 900_000_000
