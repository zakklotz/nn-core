import torch

from nncore.constraints import build_constraint, register_constraint
from nncore.models import ConstraintConfig
from nncore.train import Trainer


@register_constraint("dummy_test_constraint")
class DummyConstraint:
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def compute(self, *, model, batch, outputs, step=None, state=None):
        loss = torch.tensor(self.scale, device=outputs["loss"].device)
        return {"constraint/dummy": loss}


def test_constraint_registry_build_and_compute():
    c = build_constraint("dummy_test_constraint", {"scale": 2.0})
    out = c.compute(model=None, batch={}, outputs={"loss": torch.tensor(0.0)}, step=0)

    assert "constraint/dummy" in out
    assert torch.allclose(out["constraint/dummy"], torch.tensor(2.0))


def test_trainer_constraint_loss_integration():
    torch.manual_seed(0)

    model = torch.nn.Linear(4, 3)
    model.config = type("Cfg", (), {})()
    model.config.constraints = [
        ConstraintConfig(name="dummy_test_constraint", weight=3.0, params={"scale": 2.0})
    ]

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, device="cpu", amp=False)

    x = torch.randn(8, 4)
    y = torch.randn(8, 3)

    def forward_fn(m, batch):
        xb, yb = batch
        pred = m(xb)
        base_loss = torch.nn.functional.mse_loss(pred, yb)
        return {"loss": base_loss}

    with torch.no_grad():
        base_loss = torch.nn.functional.mse_loss(model(x), y).item()

    out = trainer.train_step(forward_fn, (x, y))

    assert "constraint/dummy" in out
    assert abs(out["constraint/dummy"] - 6.0) < 1e-6
    assert abs(out["loss"] - (base_loss + 6.0)) < 1e-4
