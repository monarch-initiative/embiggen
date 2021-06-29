"""Submodule to validate parameters from user input with meaningful errors."""
from typing import Union


def validate_verbose(candidate_verbose: Union[bool, int]) -> bool:
    """Return validated verbose candidate, raising a meaningful error otherwise.

    Parameters
    -----------------
    candidate_verbose: bool,
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
