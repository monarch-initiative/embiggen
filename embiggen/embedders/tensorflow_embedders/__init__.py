"""Module with graph embedding models based on TensorFlow."""
from .tensorflow_embedder import TensorFlowEmbedder
from ...utils import build_init

build_init(
    module_library_name="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=TensorFlowEmbedder
)