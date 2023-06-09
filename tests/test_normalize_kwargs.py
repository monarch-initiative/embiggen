"""Validates that all model parameters can be normalized"""

from embiggen import get_available_models_for_node_embedding, get_available_models_for_edge_prediction
from embiggen.utils.normalize_kwargs import normalize_kwargs

from embiggen.utils.abstract_models.abstract_embedding_model import AbstractEmbeddingModel
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


def test_normalize_kwargs_for_node_embedding_models():
    """Test that all node embedding models parameters can be normalized."""
    df = get_available_models_for_node_embedding()

    for _, row in df.iterrows():
        model = AbstractEmbeddingModel.get_model_from_library(
            model_name=row.model_name,
            task_name="Node Embedding",
            library_name=row.library_name,
        )()

        AbstractEmbeddingModel.get_model_from_library(
            model_name=row.model_name,
            task_name="Node Embedding",
            library_name=row.library_name
        )(**normalize_kwargs(model, model.parameters()))

        AbstractEmbeddingModel.get_model_from_library(
            model_name=row.model_name,
            task_name="Node Embedding",
            library_name=row.library_name
        )(**normalize_kwargs(model, model.smoke_test_parameters()))


def test_normalize_kwargs_for_link_prediction_models():
    """Test that all link prediction models parameters can be normalized."""
    df = get_available_models_for_edge_prediction()

    for _, row in df.iterrows():
        model = AbstractEdgePredictionModel.get_model_from_library(
            model_name=row.model_name,
            task_name=AbstractEdgePredictionModel.task_name(),
            library_name=row.library_name
        )()

        AbstractEdgePredictionModel.get_model_from_library(
            model_name=row.model_name,
            task_name=AbstractEdgePredictionModel.task_name(),
            library_name=row.library_name
        )(**normalize_kwargs(model, model.parameters()))

        AbstractEdgePredictionModel.get_model_from_library(
            model_name=row.model_name,
            task_name=AbstractEdgePredictionModel.task_name(),
            library_name=row.library_name
        )(**normalize_kwargs(model, model.smoke_test_parameters()))
        