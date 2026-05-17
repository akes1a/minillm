"""v0 stage test: overfit a tiny fixed token batch.

This script is intentionally separate from the real training entry point. It is a
small diagnostic for the MVP model backbone: if the model cannot memorize one
fixed batch, the training path has a bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minillm.core.config import ModelConfig  # noqa: E402
from minillm.core.model import MiniLLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v0 tiny overfit stage test.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--min-loss-drop", type=float, default=0.75)
    parser.add_argument("--max-final-loss", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(device)


def make_fixed_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = torch.arange(seq_len, device=device)
    offsets = torch.arange(batch_size, device=device).unsqueeze(1) * 7
    return (base.unsqueeze(0) + offsets) % vocab_size


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        dropout=0.0,
    )
    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    input_ids = make_fixed_batch(args.batch_size, args.seq_len, args.vocab_size, device)

    initial_loss = None
    final_loss = None

    print(f"device={device}")
    print(f"parameters={model.num_parameters():,}")

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids)
        if output.loss is None:
            raise RuntimeError("model returned no loss")

        loss = output.loss
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss_value:.6f}")

    assert initial_loss is not None
    assert final_loss is not None

    prompt = input_ids[:1, :4]
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    print(f"prompt={prompt.detach().cpu().tolist()[0]}")
    print(f"generated={generated.detach().cpu().tolist()[0]}")

    loss_drop = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    print(f"initial_loss={initial_loss:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"loss_drop={loss_drop:.2%}")

    if loss_drop < args.min_loss_drop:
        print(f"FAILED: loss_drop < {args.min_loss_drop:.2%}", file=sys.stderr)
        return 1
    if final_loss > args.max_final_loss:
        print(f"FAILED: final_loss > {args.max_final_loss}", file=sys.stderr)
        return 1

    print("PASSED: tiny overfit stage test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

