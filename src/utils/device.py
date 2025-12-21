import torch


def get_device():
    """
    Returns the best available device in priority order:
    1) MPS (Apple Silicon)
    2) CUDA
    3) CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
