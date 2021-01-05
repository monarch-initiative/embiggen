import numpy as np
import pandas as pd
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from ensmallen_graph import EnsmallenGraph
from typing import Dict
from .layers import edge_embedding_layer
from ..embedders import Embedder
from ..sequences import LinkPredictionSequence
from extra_keras_metrics import get_standard_binary_metrics


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
        batches_per_epoch: int = 2**10,
        validation_batches_per_epoch: int = 2**8,
        negative_samples: float = 1.0,
        epochs: int = 100,
        validation_split: float = 0.2,
        patience: int = 5,
        min_delta: float = 0.0001,
        monitor: str = "val_loss",
        mode: str = "min",
        random_state: int = 42,
        fast_sampling: bool = True,
        verbose: bool = True,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        """Train model and return training history dataframe.

        Parameters
        -------------------
        graph: EnsmallenGraph,
            Graph object to use for training.
        batch_size: int = 2**18,
            Batch size for the training process.
        batches_per_epoch: int = 2**10,
            Number of batches to train for in each epoch.
        validation_batches_per_epoch: int = 2**8,
            Validation batches per epoch.
        negative_samples: float = 1.0,
            Rate of unbalancing in the batch.
        epochs: int = 100,
            Number of epochs to train for.
        validation_split: float = 0.2,
            Split to use for validation set and early stopping.
        patience: int = 5,
            How many epochs to wait for before stopping the training.
        min_delta: float = 0.0001,
            Minimum variation of loss function.
        monitor: str = "val_loss",
            Metric to monitor for early stopping.
        mode: str = "min",
            Direction of the minimum delta.
        random_state: int = 42,
            Random state to use for validation split.
        fast_sampling: bool = True,
            Wether to enable the fast sampling.
            You may want to disable this when using very big graphs
            that may not fit into main memory.
        verbose: bool = True,
            Wether to show loading bars.
        ** kwargs: Dict,
            Parameteres to pass to the dit call.

        Returns
        --------------------
        Dataframe with traininhg history.
        """
        train_graph, validation_graph = graph.connected_holdout(
            train_size=1-validation_split,
            random_state=random_state,
            verbose=verbose
        )
        if fast_sampling:
            train_graph.enable(
                vector_sources=True,
                vector_destinations=True,
            )
            validation_graph.enable(
                vector_sources=True,
                vector_destinations=True,
            )
        training_sequence = LinkPredictionSequence(
            train_graph,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples,
        )
        validation_sequence = LinkPredictionSequence(
            validation_graph,
            method=None,
            batch_size=batch_size,
            batches_per_epoch=validation_batches_per_epoch,
            negative_samples=negative_samples,
            # We need to avoid sampling edges from the training graph
            # as negatives.
            graph_to_avoid=train_graph
        )
        return super().fit(
            training_sequence,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_sequence,
            callbacks=[
                EarlyStopping(
                    monitor=monitor,
                    min_delta=min_delta,
                    patience=patience,
                    mode=mode,
                    restore_best_weights=True
                )
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
