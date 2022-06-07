"""Submodule providing utilities to normalize parameters for modular models."""
from typing import List, Any, Union, Type


def normalize_model_list_parameter(
    candidate_list: Union[Any, List[Any]],
    elements_number: int,
    object_type: Type,
    default_value: Any = None,
    can_be_empty: bool = False
) -> List[Any]:
    """Return given candidate list parameter normalized.

    Parameters
    --------------------------
    candidate_list: Union[Any, List[Any]],
        Candidate list to be validated and normalized.
    elements_number: int,
        Number of expand to expand the list to or expected list size.
    object_type: Type,
        Base object of the list.
    default_value: Any = None,
        The type to use when the provided candidate list object is None.
    can_be_empty: bool = False
        Whether the list can be empty.

    Raises
    --------------------------
    TypeError,
        If the given candidate list base type is not compatible with the provided object type.
    ValueError,
        If the given number of elements in the list does not match the provided value.
    ValueError,
        If the number of elements provided is not a strictly positive integer.

    Returns
    --------------------------
    Validated and normalized list of parameters.
    """
    # Validate the provided number of elements
    if not isinstance(elements_number, int) or not can_be_empty and elements_number < 1:
        raise ValueError(
            (
                "The provided amount of elements is not strictly positive "
                "integer value. Namely, the provided object was `{}`."
            ).format(elements_number)
        )

    # Use the default value if the provided one is None.
    if candidate_list is None:
        candidate_list = default_value

    # Normalize the provided candidate list to a list
    if not isinstance(candidate_list, list):
        # Verify that the value in the list is effectively of the expected type
        if not isinstance(candidate_list, object_type):
            raise ValueError(
                (
                    "Object in candidate list was expected to be "
                    "of type `{}`, but object of type `{}` was found."
                ).format(
                    object_type,
                    type(candidate_list)
                )
            )
        # Expand the scalar to a list
        candidate_list = [
            candidate_list
            for _ in range(elements_number)
        ]

    # Validate the shape of the list
    # This is a tautological step when the provided element
    # was a scalar and expanded at the previous step.
    if len(candidate_list) != elements_number:
        raise ValueError(
            (
                "The provided candidate list length `{}` does not "
                "match the expected candidate list length `{}`."
            ).format(
                len(candidate_list),
                elements_number
            )
        )

    # Check that all elements in the list match with the
    # expected type.
    for element in candidate_list:
        # Verify that the element in the list is effectively of the expected type
        if not isinstance(element, object_type):
            raise ValueError(
                (
                    "Object `{}` in candidate list was expected to be "
                    "of type `{}`, but object of type `{}` was found."
                ).format(
                    element,
                    object_type,
                    type(element)
                )
            )

    # Finally, returned the validated and sanitized candidate list
    return candidate_list


def normalize_model_ragged_list_parameter(
    candidate_ragged_list: Union[Any, List[Any], List[List[Any]]],
    submodules_number: int,
    layers_per_submodule: List[int],
    object_type: Type,
    default_value: Any = None
) -> List[List[Any]]:
    """Return validated and normalized ragged list of parameters.

    Parameters
    --------------------------
    candidate_ragged_list: Union[Any, List[Any], List[List[Any]]],
        Candidate ragged list to be normalized and validated.
    submodules_number: int,
        Number of submodules.
    layers_per_submodule: List[int]
        Dimensions for the sub-lists in the ragged list.
    object_type: Type,
        Base object of the ragged list.
    default_value: Any = None,
        The type to use when the provided candidate list object is None.

    Raises
    --------------------------
    ValueError,
        If the number of elements provided is not a strictly positive integer.

    Returns
    --------------------------
    Validated and normalized ragged list.
    """
    # Validate the layers per submodule list
    # and submodule
    normalize_model_list_parameter(
        layers_per_submodule,
        submodules_number,
        int,
        default_value
    )
    # Expand the provided ragged list if it is a scalar
    if not isinstance(candidate_ragged_list, list):
        candidate_ragged_list = normalize_model_list_parameter(
            candidate_ragged_list,
            submodules_number,
            object_type,
            default_value
        )
    # Expand the provided list of elements if it is not a list of lists
    candidate_ragged_list = [
        normalize_model_list_parameter(
            element,
            layers_number,
            object_type,
            default_value
        )
        for element, layers_number in zip(
            candidate_ragged_list,
            layers_per_submodule
        )
    ]
    # Return normalized ragged list
    return candidate_ragged_list
