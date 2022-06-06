"""Submodule providing node embedding models implemented in Karate Club."""
from ...utils import build_init, AbstractEmbeddingModel

build_init(
    module_library_names="karateclub",
    formatted_library_name="Karate Club",
    expected_parent_class=AbstractEmbeddingModel
)
