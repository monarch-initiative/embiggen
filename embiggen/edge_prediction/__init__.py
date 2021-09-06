"""Module implementing link prediction models."""
from .perceptron import Perceptron
from .multi_layer_perceptron import MultiLayerPerceptron

__all__ = [
    "Perceptron",
    "MultiLayerPerceptron"
]
