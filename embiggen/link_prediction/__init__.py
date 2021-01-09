"""Module implementing link prediction models."""
from .perceptron import Perceptron
from .degree_perceptron import DegreePerceptron
from .multi_layer_perceptron import MultiLayerPerceptron

__all__ = [
    "Perceptron",
    "DegreePerceptron",
    "MultiLayerPerceptron"
]
