import torch

from nncore.models import Transformer
from nncore.train import Trainer
from nncore.utils import get_device, describe_device, set_seed
from nncore.smoke import ToyLMConfig, make_toy_lm_batch, toy_lm_forward_fn


def main():
    set_seed(0)

    device = get_device()
    print(f"Using device: {describe_device(device)}")

    # Small model so it runs fast on CPU or GPU
    cfg = ToyLMConfig(vocab_size=128, seq_len=64, batch_size=32)

    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=128,
        num_heads=8,
        max_seq_len=cfg.seq_len,
        num_encoder_layers=0,
        num_decoder_layers=4,
        norm_style="pre",
        attn_backend="manual",   # try "sdpa" too
        tie_weights=True,
        return_hidden=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = Trainer(model, optimizer, device=device, amp=(device.type == "cuda"))

    print("Running toy LM training...")
    losses = []
    for step in range(30):
        batch = make_toy_lm_batch(cfg, device=device)
        out = trainer.train_step(toy_lm_forward_fn, batch)
        losses.append(out["loss"])
        if (step + 1) % 5 == 0:
            print(f"step {step+1:03d} | loss {out['loss']:.4f} | stepped={out['stepped']}")

    print(f"loss start={losses[0]:.4f} end={losses[-1]:.4f}")
    print("Toy LM smoke complete ✔")


if __name__ == "__main__":
    main()
