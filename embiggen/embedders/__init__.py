"""Submodule providing TensorFlow and Ensmallen-based embedders."""
from embiggen.embedders.ensmallen_embedders import *
from embiggen.embedders.tensorflow_embedders import *
from embiggen.embedders.pykeen_embedders import *
from embiggen.embedders.non_existent_embedders import *
from embiggen.embedders.karateclub_embedders import *
from embiggen.embedders.graph_embedding_pipeline import embed_graph

# Export all non-internals.
__all__ = [
    variable_name
    for variable_name in locals().keys()
    if not variable_name.startswith("_")
]
