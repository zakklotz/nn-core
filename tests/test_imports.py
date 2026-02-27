import torch
import nncore
from nncore.utils import (
    get_device,
    get_logger,
    set_seed,
)
from nncore.io import save_checkpoint, load_checkpoint


def test_imports():
    # Version exists
    assert hasattr(nncore, "__version__")

    # Device utilities work without crashing
    device = get_device(prefer="cpu")
    assert device.type == "cpu"

    # Logger can be created
    logger = get_logger()
    assert logger is not None

    # Seed runs without error
    set_seed(123)

    # Check checkpoint functions exist
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
