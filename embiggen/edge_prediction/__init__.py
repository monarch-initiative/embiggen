"""Submodule providing models for edge prediction."""
from .edge_prediction_sklearn import SklearnModelEdgePredictionAdapter


try:
    from .edge_prediction_tensorflow import (
        Perceptron, MultiLayerPerceptron, EdgePredictionGraphNeuralNetwork,
        FeedForwardNeuralNetwork, EdgePredictionModel
    )

    tensorflow_edge_prediction_models = {
        "perceptron": Perceptron,
        "multilayerperceptron": MultiLayerPerceptron,
    }

    def is_tensorflow_edge_prediction_method(model_name: str) -> bool:
        return model_name.lower() in tensorflow_edge_prediction_models

    def get_tensorflow_model(model_name: str, **kwargs) -> EdgePredictionModel:
        return tensorflow_edge_prediction_models[model_name.lower()](**kwargs)

    __all__ = [
        "SklearnModelEdgePredictionAdapter",
        "Perceptron",
        "MultiLayerPerceptron",
        "EdgePredictionGraphNeuralNetwork",
        "FeedForwardNeuralNetwork",
        "get_tensorflow_model"
    ]
except ModuleNotFoundError as e:
    def is_tensorflow_edge_prediction_method(model_name: str) -> bool:
        return False

    def get_tensorflow_model(model_name: str, **kwargs):
        raise NotImplementedError(
            "The method is not available when TensorFlow is not installed."
        )

    __all__ = [
        "SklearnModelEdgePredictionAdapter",
        "get_tensorflow_model"
    ]
