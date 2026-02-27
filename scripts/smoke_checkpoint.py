import torch
from pathlib import Path

from nncore.utils import get_device, set_seed, describe_device
from nncore.io import save_checkpoint, load_checkpoint


def main():
    set_seed(123)

    device = get_device()
    print(f"Using device: {describe_device(device)}")

    # Tiny model
    model1 = torch.nn.Linear(4, 3).to(device)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    # One training step
    model1.train()
    opt1.zero_grad(set_to_none=True)

    x = torch.randn(8, 4, device=device)
    y = torch.randn(8, 3, device=device)

    pred = model1(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt1.step()

    # Save checkpoint
    ckpt_path = Path("smoke_ckpt.pt")
    save_checkpoint(
        str(ckpt_path),
        model=model1,
        optimizer=opt1,
        step=1,
        epoch=0,
        extra={"smoke": True},
    )

    # Load into new model
    model2 = torch.nn.Linear(4, 3).to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    load_checkpoint(str(ckpt_path), model=model2, optimizer=opt2)

    # Verify parameters match
    for k, v in model1.state_dict().items():
        if not torch.allclose(v, model2.state_dict()[k]):
            raise RuntimeError(f"Mismatch in parameter: {k}")

    print("Checkpoint roundtrip successful ✔")


if __name__ == "__main__":
    main()