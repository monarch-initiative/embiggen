from typing import Dict

import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from extra_keras_metrics import get_standard_binary_metrics
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from ..embedders import Embedder
from ..sequences import LinkPredictionSequence
from .layers import edge_embedding_layer


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
        self._model_name = self.__class__.__name__
        super().__init__(*embedding.shape, optimizer)
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=self._optimizer,
            metrics=get_standard_binary_metrics(),
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
            name="{}_{}".format(
                self._model_name,
                self._edge_embedding_method
            )
        )

    def fit(
        self,
        graph: EnsmallenGraph,
        batch_size: int = 2**18,
        batches_per_epoch: int = 2**14,
        negative_samples: float = 1.0,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        """Train model and return training history dataframe.

        Parameters
        -------------------
        graph: EnsmallenGraph,
            Graph object to use for training.
        batch_size: int = 2**18,
            Batch size for the training process.
        batches_per_epoch: int = 2**14,
            Number of batches to train for in each epoch.
        negative_samples: float = 1.0,
            Rate of unbalancing in the batch.
        ** kwargs: Dict,
            Parameteres to pass to the dit call.

        Returns
        --------------------
        Dataframe with traininhg history.
        """
        sequence = LinkPredictionSequence(
            graph,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples,
        )
        return super().fit(
            sequence,
            **kwargs
        )

    def predict(self, *args, **kwargs):
        """Run predict."""
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Run predict."""
        return dict(zip(
            self._model.metrics_names,
            self._model.evaluate(*args, **kwargs)
        ))
