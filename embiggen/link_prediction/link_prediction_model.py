import numpy as np
import pandas as pd
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from ensmallen_graph import EnsmallenGraph
from typing import Dict
from .layers import edge_embedding_layer
from ..embedders import Embedder
from ..sequences import LinkPredictionSequence
from extra_keras_metrics import get_binary_metrics


class LinkPredictionModel(Embedder):

    def __init__(
        self,
        embedding: np.ndarray,
        edge_embedding_method: str = "Hadamard",
        optimizer: str = "nadam",
        trainable: bool = True
    ):
        """Create new Perceptron object."""
        if edge_embedding_method not in edge_embedding_layer:
            raise ValueError(
                "The given edge embedding method `{}` is not supported.".format(
                    edge_embedding_method
                )
            )
        self._edge_embedding_method = edge_embedding_method
        self._embedding = embedding
        self._trainable = trainable
        super().__init__(*embedding.shape, optimizer)
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=self._optimizer,
            metrics=get_binary_metrics(),
        )

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for link prediction."""
        raise NotImplementedError(
            "The method _build_model_body must be implemented in the child classes."
        )

    def _build_model(self) -> Model:
        """Build new model for link prediction."""
        embedding_layer = edge_embedding_layer[self._edge_embedding_method](
            embedding=self._embedding,
            trainable=self._trainable
        )

        return Model(
            inputs=embedding_layer.inputs,
            outputs=self._build_model_body(embedding_layer(None)),
            name="Perceptron_{}".format(self._edge_embedding_method)
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

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)
