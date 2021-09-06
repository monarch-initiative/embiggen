"""Submodule with utilities relative to GPU availability."""
from typing import List
import shutil

import tensorflow as tf


def get_available_gpus() -> List[str]:
    """Return list with IDs of available GPU devices."""
    return tf.config.experimental.list_physical_devices('GPU')


def command_is_available(command_name: str) -> bool:
    """Return whether given bash command is available in PATH.

    Parameters
    ------------------
    command_name: str,
        The command to check availability for.

    Returns
    ------------------
    Boolean representing if the command is available in PATH.
    """
    return shutil.which(command_name) is not None


def has_nvidia_drivers() -> bool:
    """Return whether NVIDIA drivers can be detected."""
    return command_is_available("nvidia-smi")


def has_rocm_drivers() -> bool:
    """Return whether ROCM drivers can be detected."""
    return command_is_available("rocm-smi")


def get_available_gpus_number() -> int:
    """Return whether GPUs can be detected."""
    return len(get_available_gpus())


def has_single_gpu() -> bool:
    """Return whether there is only a GPU available."""
    return get_available_gpus_number() == 1


def has_gpus() -> bool:
    """Return whether GPUs can be detected."""
    return get_available_gpus_number() > 0
