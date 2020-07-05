from .embedder import Embedder
from tensorflow.keras.layers import Flatten, Input, Embedding, Dot, Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model


class SkipGram(Embedder):

    def _build_model(self):
        """Create new SkipGram model."""

        # Creating the input layers
        input_layers = [
            Input(shape=(1,), name=Embedder.EMBEDDING_LAYER_NAME),
            Input(shape=(1,), name="context")
        ]

        # Creating dot product of embedding layers for each input layer
        dot_product_layer = Flatten()(Dot(axes=2)([
            Embedding(
                self._vocabulary_size,
                self._embedding_size,
                input_length=1
            )(input_layer)
            for input_layer in input_layers
        ]))

        # Output layer from previous dot product
        output = Dense(1, activation="sigmoid")(dot_product_layer)

        # Creating the actual model
        skipgram = Model(
            inputs=input_layers,
            outputs=output,
            name="SkipGram"
        )

        # Compiling the model
        skipgram.compile(
            loss='binary_crossentropy',
            optimizer=self._optimizer,
            metrics=[
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc"),
                Precision(),
                Recall(),
            ]
        )

        return skipgram
