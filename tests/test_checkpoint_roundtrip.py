import torch
from nncore.io import save_checkpoint, load_checkpoint
from nncore.utils import set_seed

def test_checkpoint_roundtrip(tmp_path):
    set_seed(123)

    # Build a tiny model + optimizer
    model1 = torch.nn.Linear(4, 3)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    # One training step (CPU)
    model1.train()
    opt1.zero_grad(set_to_none=True)
    x = torch.randn(8, 4)
    y = torch.randn(8, 3)
    pred = model1(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt1.step()

    # Save checkpoint
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        str(ckpt_path),
        model=model1,
        optimizer=opt1,
        step=1,
        epoch=0,
        extra={"note": "roundtrip"},
    )

    # Load into fresh model + optimizer
    model2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(str(ckpt_path), model=model2, optimizer=opt2)

    # Assert params match (tensor-by-tensor)
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1.keys():
        assert torch.allclose(sd1[k], sd2[k])

    # Sanity check metadata preserved
    assert ckpt["meta"]["step"] == 1
    assert ckpt["meta"]["epoch"] == 0
    assert ckpt["extra"]["note"] == "roundtrip"
