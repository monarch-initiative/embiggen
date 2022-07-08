"""Submodule to test whether the metaprogramming is working properly."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names="non_existent_module",
    formatted_library_name="Non Existent Module",
    expected_parent_class=AbstractEmbeddingModel
)
