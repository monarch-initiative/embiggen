"""Module with node embedding models based on PyKEEN."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names=["torch", "pykeen"],
    formatted_library_name="PyKEEN",
    task_name="Node Embedding",
    expected_parent_class=AbstractEmbeddingModel
)