"""This module contains the normalize_kwargs method."""
from typing import Any, Dict, Union, List, Type, Tuple
import compress_json
from embiggen.utils.abstract_models import AbstractModel


def get_data_type_from_data_type_name(data_type_name: Union[str, List[str]]) -> List[Type]:
    """Returns the data type from the data type name.
    
    Parameters
    ----------
    data_type_name : str
        The name of the data type.
    """
    if isinstance(data_type_name, str):
        data_type_name = [data_type_name]
    
    data_types = []
    for dtn in data_type_name:
        if dtn == "bool":
            data_types.append(bool)
        if dtn == "str":
            data_types.append(str)
        if dtn == "None":
            data_types.append(type(None))
        if dtn == "int":
            data_types.append(int)
        if dtn == "float":
            data_types.append(float)
    
    if len(data_types) == 0:
        raise NotImplementedError(
            f"The provided data type name {data_type_name} is not supported. "
            "Please open an issue on the Embigggen repository "
            "detailing the problem."
        )
    return data_types


def normalize_object_from_data_type_name(data_type_name: str, value: Any) -> Any:
    """Returns the normalized object from the data type name.
    
    Parameters
    ----------
    data_type_name : str
        The name of the data type.
    value : Any
        The value to normalize.
    """
    if isinstance(data_type_name, str):
        data_type_name = [data_type_name]
    
    for dtn in data_type_name:
        if dtn == "bool":
            return bool(value)
        if dtn == "int":
            try:
                return int(value)
            except ValueError:
                pass
        if dtn == "float":
            return float(value)
        if dtn == "str":
            return str(value)
    
    raise NotImplementedError(
        f"The provided data type name {data_type_name} is not supported "
        f"for the value {value}. "
        "Please open an issue on the Embigggen repository "
        "detailing the problem."
    )


def normalize_kwargs(
    model: Type[AbstractModel],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Returns normalized kwargs.

    Implementation details
    ----------------------
    This method is necessary to handle corner cases such as exotic
    data types that can happen when loading for instance a JSON file
    or a Pandas DataFrame. For instance:

    * A bool when loaded in pandas may assume the datatype 'bool_'.
    * An integer value when loaded in pandas may assume the datatype 'float32' or 'float64'.

    """
    normalization_schemas = compress_json.local_load("normalization_schemas.json", use_cache=True)

    unsupported_parameters = []

    for key, value in kwargs.items():
        if key not in normalization_schemas:
            unsupported_parameters.append(key)
            continue

        # We retrieve the expected type for the given parameter.
        expected_type = normalization_schemas[key]

        # If this is a special case where we can skip the normalization,
        # we do so.
        if expected_type == "pass":
            continue

        # We retrieve the type of the value and we check if it is the same
        # as the expected type.
        valid_types: Tuple[Type] = tuple(get_data_type_from_data_type_name(expected_type))
        if isinstance(value, valid_types):
            # The type is already correct, we can skip the normalization.
            continue
        
        try:
            # If the type is not correct, we try to convert it.
            kwargs[key] = normalize_object_from_data_type_name(expected_type, value)
        except (TypeError, ValueError) as exception:
            # If the type cannot be converted, we raise an error.
            raise TypeError(
                "While we were normalizing the parameters, we found an error. "
                f"The parameter {key} has the value \"{value}\" with type {type(value)} "
                f"but the expected type is {expected_type}. "
                f"The model is {model.model_name()} from library {model.library_name()} "
                f"for the task {model.task_name()}."
            ) from exception

    if len(unsupported_parameters) > 0:
        raise NotImplementedError(
            f"The following parameters are not supported: {unsupported_parameters}. "
            "Please open an issue on the Embigggen repository "
            "detailing the problem. "
            f"The model is {model.model_name()} from library {model.library_name()} "
            f"for the task {model.task_name()}."
        )

    return kwargs