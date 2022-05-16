"""Module with graph embedding models based on TensorFlow."""
from .pykeen_embedder import PyKeenEmbedder
from ...utils import build_init

build_init(
    module_library_names=["torch", "pykeen"],
    formatted_library_name="PyKeen",
    expected_parent_class=PyKeenEmbedder
)