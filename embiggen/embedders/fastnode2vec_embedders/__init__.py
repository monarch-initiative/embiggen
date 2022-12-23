"""Module with node embedding models based on FastNode2Vec."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names=["numba", "fastnode2vec"],
    formatted_library_name="FastNode2Vec",
    expected_parent_class=AbstractEmbeddingModel
)