# tests/test_training_step.py

import torch

from nncore.train import Trainer


def test_training_step_updates_weights():
    torch.manual_seed(0)

    model = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(model, opt, device="cpu", amp=False)

    x = torch.randn(16, 4)
    y = torch.randn(16, 3)

    def forward_fn(m, batch):
        xb, yb = batch
        pred = m(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        return {"loss": loss}

    w_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    out1 = trainer.train_step(forward_fn, (x, y))
    out2 = trainer.train_step(forward_fn, (x, y))

    w_after = model.state_dict()

    # At least one param tensor should have changed
    changed = False
    for k in w_before.keys():
        if not torch.allclose(w_before[k], w_after[k]):
            changed = True
            break

    assert changed
    assert isinstance(out1["loss"], float)
    assert isinstance(out2["loss"], float)