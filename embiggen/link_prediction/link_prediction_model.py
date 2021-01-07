from typing import Dict

import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from extra_keras_metrics import get_standard_binary_metrics
from tensorflow.keras.callbacks import EarlyStopping
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
        batch_size: int = 2**16,
        batches_per_epoch: int = 2**8,
        negative_samples: float = 1.0,
        max_epochs: int = 1000,
        validation_split: float = 0.2,
        random_state: int = 42,
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = "min",
        monitor: str = "val_loss",
        verbose: bool = True,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        """Train model and return training history dataframe.

        Parameters
        -------------------
        graph: EnsmallenGraph,
            Graph object to use for training.
        batch_size: int = 2**16,
            Batch size for the training process.
        batches_per_epoch: int = 2**8,
            Number of batches to train for in each epoch.
        negative_samples: float = 1.0,
            Rate of unbalancing in the batch.
        max_epochs: int = 1000,
            Maximal number of epochs to train for.
        validation_split: float = 0.2,
            Percentage of split for validation.
        random_state: int = 42,
            Random state to use for the split.
        patience: int = 5,
            Number of epochs for wait for to trigger early stopping.
        min_delta: float = 0.0001,
            Minimum variation of delta.
        mode: str = "min",
            Direction of the early stopping delta.
        monitor: str = "val_loss",
            Metric to monitor.
        verbose: bool = True,
            Wether to shop loading bars.
        ** kwargs: Dict,
            Parameteres to pass to the dit call.

        Returns
        --------------------
        Dataframe with traininhg history.
        """
        train, validation = graph.connected_holdout(
            validation_split,
            random_state=random_state,
            verbose=verbose
        )
        training_sequence = LinkPredictionSequence(
            train,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples,
        )
        validation_sequence = LinkPredictionSequence(
            validation,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples,
            graph_to_avoid=train
        )
        return super().fit(
            training_sequence,
            epochs=max_epochs,
            validation_data=validation_sequence,
            verbose=verbose,
            callbacks=[
                EarlyStopping(
                    monitor=monitor,
                    min_delta=min_delta,
                    patience=patience,
                    mode=mode,
                    restore_best_weights=True
                ),
                *kwargs.get("callbacks", [])
            ],
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
