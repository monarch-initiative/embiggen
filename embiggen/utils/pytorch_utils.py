"""Submodule with utilities for pytorch models."""
import torch

def validate_torch_device(device: str) -> str:
    """Validate and sanitize torch device name.
    
    Parameters
    ------------------------
    device: str
        The device to use with Torch.
        This can either be `cuda`, `gpu`, or `auto`.

    Raises
    ------------------------
    ValueError
        If the device is not either cpu or cuda.
    ValueError
        If cuda was requested but CUDA is not available.
    """

    if torch.cuda.is_available():
        cuda_comment = (
            "Your Torch installation does detect CUDA "
            "installed in your system. Do consider using `cuda` "
            "as device option as it may be the faster option."
        )
        if device == "auto":
            device = "cuda"
    else:
        cuda_comment = (
            "Your Torch installation is not "
            "currently able to detect any CUDA device."
        )
        if device == "auto":
            device = "cpu"
    
    if device not in ("cpu", "cuda"):
        raise ValueError(
            f"The provided torch device `{device}` is not a supported "
            "torch device. Currently, the supported torch devices are "
            f"cpu and cuda, or `auto` for auto-dispatching. {cuda_comment}"
        )
        
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"{cuda_comment} You have provided as device `cuda`. "
            "Either use `cpu` or ensure you have a working GPU "
            "and CUDA is installed and has a version compatible "
            "with the version of Torch you have installed."
        )
    
    return device
