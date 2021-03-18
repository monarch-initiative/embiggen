"""Class for creating a MultiLayerPerceptron model for edge prediction tasks."""
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Layer

from .edge_prediction_model import EdgePredictionModel


class MultiLayerPerceptron(EdgePredictionModel):

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for edge prediction."""
        input_layer = Dense(
            units=32,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(input_layer)
        input_layer = Dense(
            units=32,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(input_layer)
        return Dense(
            units=1,
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(input_layer)
