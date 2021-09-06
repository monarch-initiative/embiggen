"""Submodule with utilities on TensorFlow versions."""
import tensorflow as tf
from packaging import version
from validate_version_code import validate_version_code


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
