"""Class for creating an FFNN model for link prediction tasks."""
from tensorflow.keras.layers import Dense, Layer, BatchNormalization, Activation, Dropout
from .link_prediction_model import LinkPredictionModel


class FFNN(LinkPredictionModel):

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for link prediction."""
        for units in (256, 128, 64):
            input_layer = Dense(units, activation="relu")(input_layer)
            input_layer = Dense(units)(input_layer)
            input_layer = BatchNormalization()(input_layer)
            input_layer = Activation(activation="relu")(input_layer)

        input_layer = Dropout(0.5)(input_layer)
        return Dense(1, activation="sigmoid")(input_layer)
