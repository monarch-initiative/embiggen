"""Validates that all model parameters can be normalized"""

from embiggen import get_available_models_for_node_embedding
from embiggen.utils.normalize_kwargs import normalize_kwargs

from embiggen.utils.abstract_models.abstract_embedding_model import AbstractEmbeddingModel


def test_normalize_kwargs():
    """Test that all models parameters can be normalized."""
    df = get_available_models_for_node_embedding()

    for _, row in df.iterrows():
        model = AbstractEmbeddingModel.get_model_from_library(
            model_name=row.model_name,
            task_name=AbstractEmbeddingModel.task_name(),
            library_name=row.library_name
        )()

        normalize_kwargs(model.parameters())
        normalize_kwargs(model.smoke_test_parameters())
