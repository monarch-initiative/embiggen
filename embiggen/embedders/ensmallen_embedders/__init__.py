"""Submodule providing node embedding models implemented in Ensmallen in Rust."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names="ensmallen",
    formatted_library_name="Ensmallen",
    expected_parent_class=AbstractEmbeddingModel
)
