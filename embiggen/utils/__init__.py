"""Sub-module with utilities to make experiments easier to write."""
from .parameter_validators import validate_verbose, validate_window_size
from .gpu_utilities import has_gpus, has_nvidia_drivers, has_rocm_drivers, has_single_gpu, execute_gpu_checks, get_available_gpus_number
from .normalize_model_structural_parameters import normalize_model_list_parameter, normalize_model_ragged_list_parameter

try:
    from .tensorflow_utils import (
        tensorflow_version_is_higher_or_equal_than,
        must_have_tensorflow_version_higher_or_equal_than
    )
    from .graph_to_sparse_tensor import graph_to_sparse_tensor
    __all__ = [
        "graph_to_sparse_tensor",
        "validate_verbose",
        "validate_window_size",
        "has_gpus",
        "has_nvidia_drivers",
        "has_rocm_drivers",
        "has_single_gpu",
        "execute_gpu_checks",
        "get_available_gpus_number",
        "tensorflow_version_is_higher_or_equal_than",
        "must_have_tensorflow_version_higher_or_equal_than",
        "normalize_model_list_parameter", "normalize_model_ragged_list_parameter"
    ]
except ModuleNotFoundError:
    __all__ = [
        "validate_verbose",
        "validate_window_size",
        "has_gpus",
        "has_nvidia_drivers",
        "has_rocm_drivers",
        "has_single_gpu",
        "execute_gpu_checks",
        "get_available_gpus_number",
        "normalize_model_list_parameter", "normalize_model_ragged_list_parameter"
    ]
