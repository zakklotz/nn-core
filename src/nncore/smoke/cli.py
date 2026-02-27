import argparse
import torch

from nncore.models import Transformer
from nncore.train import Trainer
from nncore.utils import get_device, describe_device, set_seed, EMAMeter, timer
from nncore.smoke import ToyLMConfig, make_toy_lm_batch, toy_lm_forward_fn


def main():
    parser = argparse.ArgumentParser(description="Toy LM training CLI (nn-core)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="manual")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(0)

    device = get_device(prefer=args.device)
    print(f"Using device: {describe_device(device)}")

    cfg = ToyLMConfig(
        vocab_size=128,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=args.d_model,
        num_heads=args.heads,
        max_seq_len=cfg.seq_len,
        num_encoder_layers=0,
        num_decoder_layers=args.layers,
        norm_style="pre",
        attn_backend=args.backend,
        tie_weights=True,
        return_hidden=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = Trainer(model, optimizer, device=device, amp=(device.type == "cuda"))

    loss_meter = EMAMeter(beta=0.95)

    print("Starting toy LM training...\n")

    total_tokens = 0
    with timer() as t:
        for step in range(args.steps):
            batch = make_toy_lm_batch(cfg, device=device)
            out = trainer.train_step(toy_lm_forward_fn, batch)

            loss_meter.update(out["loss"])
            total_tokens += cfg.batch_size * (cfg.seq_len - 1)

            if (step + 1) % 10 == 0:
                elapsed = t()
                tok_per_sec = total_tokens / max(elapsed, 1e-6)
                print(
                    f"step {step+1:04d} | "
                    f"ema_loss {loss_meter.avg:.4f} | "
                    f"tok/s {tok_per_sec:,.0f}"
                )

    print("\nToy LM CLI run complete ✔")


if __name__ == "__main__":
    main()
