import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from ensmallen_graph import EnsmallenGraph
from typing import Dict
from .layers import edge_embedding_layer
from ..sequences import LinkPredictionSequence


class Perceptron:

    def __init__(
        self,
        embedding: np.ndarray,
        edge_embedding_method: str = "Hadamard",
        trainable: bool = True
    ):
        """Create new Perceptron object."""
        if edge_embedding_method not in edge_embedding_layer:
            raise ValueError(
                "The given edge embedding method `{}` is not supported.".format(
                    edge_embedding_method
                )
            )
        embedding_layer = edge_embedding_layer[edge_embedding_method](
            embedding=embedding,
            trainable=trainable
        )

        self._output = Dense(1, activation="sigmoid")(embedding_layer(None))

        self._model = Model(
            inputs=embedding_layer.inputs,
            outputs=self._output,
            name="Perceptron_{}".format(edge_embedding_method)
        )

    def compile(self, optimizer: str = "nadam", **kwargs):
        return self._model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc"),
                "accuracy"
            ],
            **kwargs
        )

    def fit(
        self,
        graph: EnsmallenGraph,
        batch_size: int = 2**12,
        batches_per_epoch: int = 2**12,
        negative_samples: float = 1.0,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        sequence = LinkPredictionSequence(
            graph,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples,
        )
        return pd.DataFrame(self._model.fit(
            sequence,
            **kwargs
        ).history)

    def summary(self):
        self._model.summary()

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)
