"""Models for node label prediction."""
from .node_label_prediction_feed_forward_neural_network import NodeLabelPredictionfeedForwardNeuralNetwork
from .graph_convolutional_neural_networks import GraphConvolutionalNeuralNetwork

__all__ = [
    "NodeLabelPredictionfeedForwardNeuralNetwork",
    "GraphConvolutionalNeuralNetwork"
]
