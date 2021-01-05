"""Class for creating an MLP model for link prediction tasks."""
from tensorflow.keras.layers import Dense, Layer, Dropout
from .link_prediction_model import LinkPredictionModel


class MultiLayerPerceptron(LinkPredictionModel):

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for link prediction."""
        for _ in range(3):
            input_layer = Dense(32, activation="relu")(input_layer)
            input_layer = Dropout(0.5)(input_layer)
        return Dense(1, activation="sigmoid")(input_layer)
