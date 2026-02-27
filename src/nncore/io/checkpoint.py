import os
import torch

def save_checkpoint(
    path: str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    step: int | None = None,
    epoch: int | None = None,
    extra: dict | None = None,
) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "meta": {
            "step": step,
            "epoch": epoch,
        },
        "extra": extra,
    }

    tmp = path + ".tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)

def load_checkpoint(
    path: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)

    if model is not None and checkpoint.get("model") is not None:
        model.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint
