"""Module with graph embedding models based on TensorFlow."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractEmbeddingModel
)