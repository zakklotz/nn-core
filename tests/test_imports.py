import torch
import nncore
import os
import subprocess
import sys
from pathlib import Path
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


def test_public_tajalliyat_imports_work_from_cold_start():
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    code = (
        "from nncore.blocks import TajalliyatBlock\n"
        "from nncore.models import TajalliyatConfig, TajalliyatLM\n"
        "print(TajalliyatBlock.__name__, TajalliyatConfig.__name__, TajalliyatLM.__name__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "TajalliyatBlock TajalliyatConfig TajalliyatLM" in proc.stdout


def test_public_ofn_imports_work_from_cold_start():
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    code = (
        "from nncore.blocks import OFNBlock\n"
        "from nncore.models import OFNConfig, OFNLM\n"
        "print(OFNBlock.__name__, OFNConfig.__name__, OFNLM.__name__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "OFNBlock OFNConfig OFNLM" in proc.stdout
