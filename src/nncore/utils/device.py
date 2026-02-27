import torch

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def get_device(prefer: str = "cuda", index: int | None = None) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        if index is not None:
            return torch.device(f"cuda:{index}")
        return torch.device("cuda")
    return torch.device("cpu")

def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory // (1024**3)
        return f"{device} ({name}, {total_mem}GB)"
    return "cpu"