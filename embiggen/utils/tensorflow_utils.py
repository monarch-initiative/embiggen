"""Submodule with utilities on TensorFlow versions."""
from typing import List
import tensorflow as tf
from packaging import version
from validate_version_code import validate_version_code
from ensmallen import Graph
import warnings
import shutil


def tensorflow_version_is_higher_or_equal_than(tensorflow_version: str) -> bool:
    """Returns boolean if the TensorFlow version is higher than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.

    Returns
    ----------------------
    Boolean representing if installed TensorFlow version is higher than given one.
    """
    if not validate_version_code(tensorflow_version):
        raise ValueError(
            (
                "The provided TensorFlow version code `{}` "
                "is not a valid version code."
            ).format(tensorflow_version)
        )
    return version.parse(tf.__version__) >= version.parse(tensorflow_version)


def tensorflow_version_is_less_or_equal_than(tensorflow_version: str) -> bool:
    """Returns boolean if the TensorFlow version is less or equal than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.

    Returns
    ----------------------
    Boolean representing if installed TensorFlow version is less or equal than given one.
    """
    if not validate_version_code(tensorflow_version):
        raise ValueError(
            (
                "The provided TensorFlow version code `{}` "
                "is not a valid version code."
            ).format(tensorflow_version)
        )
    return version.parse(tf.__version__) <= version.parse(tensorflow_version)


def must_have_tensorflow_version_higher_or_equal_than(
    tensorflow_version: str,
    feature_name: str = None
):
    """Returns boolean if the TensorFlow version is higher than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.
    feature_name: str = None,
        The name of the feature that the requested version of TensorFlow
        has and previous version do not have.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.
    ValueError,
        If the installed TensorFlow version is lower than requested one.
    """
    if not tensorflow_version_is_higher_or_equal_than(tensorflow_version):
        feature_message = ""
        if feature_name is not None:
            feature_message = "\nSpecifically, the feature requested is called `{}`.".format(
                feature_name
            )
        raise ValueError(
            (
                "The required minimum TensorFlow version is `{}`, but "
                "the installed TensorFlow version is `{}`.{}"
            ).format(
                tf.__version__,
                tensorflow_version,
                feature_message
            )
        )

def get_available_gpus() -> List[str]:
    """Return list with IDs of available GPU devices."""
    try:
        import tensorflow as tf
        return tf.config.experimental.list_physical_devices('GPU')
    except ModuleNotFoundError:
        return []


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


def execute_gpu_checks():
    """Executes the GPU checks and raises the proper warnings."""
    # To avoid some nighmares we check availability of GPUs.
    if not has_gpus():
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
