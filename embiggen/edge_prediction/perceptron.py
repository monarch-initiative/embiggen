"""Class for creating a Perceptron model for link prediction tasks."""
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Layer

from .edge_prediction_model import EdgePredictionModel


class Perceptron(EdgePredictionModel):

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for edge prediction."""
        return Dense(
            units=1,
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(input_layer)
