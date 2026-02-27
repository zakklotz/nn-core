import torch

from nncore.utils import get_device, describe_device, set_seed
from nncore.train import Trainer


def main():
    set_seed(0)

    device = get_device()
    print(f"Using device: {describe_device(device)}")

    model = torch.nn.Linear(4, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    trainer = Trainer(
        model,
        optimizer,
        device=device,
        amp=(device.type == "cuda"),
    )

    x = torch.randn(256, 4, device=device)
    y = torch.randn(256, 3, device=device)

    def forward_fn(m, batch):
        xb, yb = batch
        preds = m(xb)
        loss = torch.nn.functional.mse_loss(preds, yb)
        return {"loss": loss}

    print("Running two training steps...")
    out1 = trainer.train_step(forward_fn, (x, y))
    out2 = trainer.train_step(forward_fn, (x, y))

    print(f"Loss 1: {out1['loss']:.6f}")
    print(f"Loss 2: {out2['loss']:.6f}")
    print(f"Optimizer stepped: {out2['stepped']}")


if __name__ == "__main__":
    main()