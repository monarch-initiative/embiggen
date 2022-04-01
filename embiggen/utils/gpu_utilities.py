"""Submodule with utilities relative to GPU availability."""
from typing import List
import shutil
import warnings

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


def execute_gpu_checks(use_mirrored_strategy: bool):
    """Executes the GPU checks and raises the proper warnings.
    
    Parameters
    -------------------
    use_mirrored_strategy: bool
        Whether to use the mirrored strategy.
    """
    # To avoid some nighmares we check availability of GPUs.
    if not has_gpus():
        # If there are no GPUs, mirrored strategy makes no sense.
        if use_mirrored_strategy:
            warnings.warn(
                "It does not make sense to use mirrored strategy "
                "when GPUs are not available.\n"
                "The parameter has been disabled."
            )
        # We check for drivers to try and give a more explainatory
        # warning about the absence of GPUs.
        if has_nvidia_drivers():
            warnings.warn(
                "It was not possible to detect GPUs but the system "
                "has NVIDIA drivers installed.\n"
                "It is very likely there is some mis-configuration "
                "with your TensorFlow instance.\n"
                "The model will train a LOT faster if you figure "
                "out what may be the cause of this issue on your "
                "system: sometimes a simple reboot will do a lot of good.\n"
                "If you are currently on COLAB, remember to enable require "
                "a GPU instance from the menu!"
            )
        elif has_rocm_drivers():
            warnings.warn(
                "It was not possible to detect GPUs but the system "
                "has ROCM drivers installed.\n"
                "It is very likely there is some mis-configuration "
                "with your TensorFlow instance.\n"
                "The model will train a LOT faster if you figure "
                "out what may be the cause of this issue on your "
                "system: sometimes a simple reboot will do a lot of good."
            )
        else:
            warnings.warn(
                "It was neither possible to detect GPUs nor GPU drivers "
                "of any kind on your system (neither CUDA or ROCM).\n"
                "The model will proceed with trainining, but it will be "
                "significantly slower than what would be possible "
                "with GPU acceleration."
            )
