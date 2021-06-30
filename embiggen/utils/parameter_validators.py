"""Submodule to validate parameters from user input with meaningful errors."""
from typing import Union


def validate_verbose(candidate_verbose: Union[bool, int]) -> bool:
    """Return validated verbose candidate, raising a meaningful error otherwise.

    Parameters
    -----------------
    candidate_verbose: Union[bool, int],
        The candidate verbose value to validate.

    Raises
    -----------------
    ValueError,
        If the provided verbose parameter is not within
        the allowed set of values.

    Returns
    -----------------
    The validated value.
    """
    if isinstance(candidate_verbose, bool):
        if candidate_verbose:
            candidate_verbose = 1
        else:
            candidate_verbose = 0

    if candidate_verbose not in {0, 1, 2}:
        raise ValueError(
            "Given verbose value is not valid, as it must be either "
            "a boolean value or 0, 1 or 2."
        )
    return candidate_verbose


def validate_window_size(candidate_window_size: int) -> int:
    """Return validated window size candidate, raising a meaningful error otherwise.

    Parameters
    -----------------
    candidate_window size: int,
        The candidate window size value to validate.

    Raises
    -----------------
    ValueError,
        If the provided window size parameter is not within
        the allowed set of values.

    Returns
    -----------------
    The validated value.
    """
    if not isinstance(candidate_window_size, int):
        raise ValueError(
            (
                "The window size parameter must be an integer.\n"
                "You have provided `{}`, which is of type `{}`."
            ).format(
                candidate_window_size,
                type(candidate_window_size)
            )
        )
    if candidate_window_size < 1:
        raise ValueError(
            (
                "The window size parameter must a strictly positive "
                "integer value. You have provided `{}`."
            ).format(
                candidate_window_size
            )
        )
    return candidate_window_size
