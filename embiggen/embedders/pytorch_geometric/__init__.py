"""Module with node embedding models based on PyTorch Geometric."""
from embiggen.utils.abstract_models import build_init, AbstractEmbeddingModel

build_init(
    module_library_names=["torch", "torch_geometric", "torch_cluster", "torch_sparse", "torch_scatter"],
    formatted_library_name="PyTorch Geometric",
    expected_parent_class=AbstractEmbeddingModel
)