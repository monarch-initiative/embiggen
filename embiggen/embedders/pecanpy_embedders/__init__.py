"""Module with node embedding models based on PecanPy."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names=["numba", "pecanpy"],
    formatted_library_name="PecanPy",
    task_name="Node Embedding",
    expected_parent_class=AbstractEmbeddingModel
)