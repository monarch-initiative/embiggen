from numba import types  # type: ignore
import numpy as np  # type: ignore

############################
# Types relative to nodes  #
############################

numba_nodes_type = types.int64
numba_vector_nodes_type = types.int64[:]
numpy_nodes_type = np.int64
numpy_nodes_colors_type = np.uint16
nodes_mapping_type = (types.string, numba_nodes_type)

############################
# Types relative to edges  #
############################

numba_edges_type = types.int64
numba_vector_edges_type = types.int64[:]
numpy_edges_type = np.int64
numpy_edges_colors_type = np.uint16
edges_type_list = types.ListType(numba_edges_type)

############################
# Types relative to alias  #
############################

numba_vector_alias_indices_type = types.uint16[:]
numpy_vector_alias_indices_type = np.uint16
numba_vector_alias_probs_type = types.float64[:]
numpy_vector_alias_probs_type = np.float64

alias_list_type = types.Tuple((
    numba_vector_alias_indices_type,
    types.float64[:]
))

__all__ = [
    "numba_nodes_type",
    "numba_vector_nodes_type",
    "numpy_nodes_type",
    "numba_vector_alias_indices_type",
    "numpy_alias_indices_type",
    "numba_edges_type",
    "numpy_edges_type",
    "edges_type_list",
    "float_list",
    "edges_type",
    "nodes_type",
    "alias_list_type"
]
