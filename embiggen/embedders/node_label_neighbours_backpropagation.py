"""Model implementing Node Label max_neighbours Backpropagation for graphs."""
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from extra_keras_metrics import get_minimal_multiclass_metrics
from keras_mixed_sequence import MixedSequence, VectorSequence
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import (Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from ..sequences import NodeLabelNeighboursSequence
from .embedder import Embedder


class NoLaN(Embedder):
    """Class implementing NoLaN.

    NoLaN is a Node-Label max_neighbours backpropagation model for graphs.

    """

    def __init__(
        self,
        graph: EnsmallenGraph,
        labels_number: int,
        embedding_size: int = None,
        use_dropout: bool = True,
        dropout_rate: float = 0.5,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        trainable_embedding: bool = True,
        support_mirror_strategy: bool = False,
    ):
        """Create new NoLaN model.

        Parameters
        -------------------
        graph: EnsmallenGraph,
            Graph to be embedded.
        labels_number: int,
            Number of labels.
        embedding_size: int = None,
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        use_dropout: bool = True,
            Whether to use dropout.
        dropout_rate: float = 0.5,
            Dropout rate.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        optimizer: Union[str, Optimizer] = None,
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        trainable_embedding: bool = True,
            Wether to allow for trainable embedding.
            By default true.
        support_mirror_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        """
        self._graph = graph
        self._labels_number = labels_number
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate
        self._support_mirror_strategy = support_mirror_strategy
        super().__init__(
            vocabulary_size=graph.get_nodes_number() if embedding is None else None,
            embedding_size=embedding_size if embedding is None else None,
            embedding=embedding,
            optimizer=optimizer,
            trainable_embedding=trainable_embedding
        )

    def _build_model(self):
        """Return NoLaN model."""
        neighbours_input = Input((None,), name="NeighboursInput")

        node_embedding_layer = Embedding(
            input_dim=self._vocabulary_size+1,
            output_dim=self._embedding_size,
            weights=None if self._embedding is None else [np.vstack([
                np.zeros(self._embedding_size),
                self._embedding
            ])],
            mask_zero=True,
            name=Embedder.EMBEDDING_LAYER_NAME
        )

        neighbours_embedding = GlobalAveragePooling1D()(
            node_embedding_layer(neighbours_input),
            mask=node_embedding_layer.compute_mask(neighbours_input)
        )

        if self._use_dropout:
            neighbours_embedding = Dropout(
                self._dropout_rate
            )(neighbours_embedding)

        output = Dense(
            self._labels_number,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            activation="softmax"
        )(neighbours_embedding)

        model = Model(
            inputs=neighbours_input,
            outputs=output,
            name="NodeLabelPredictor"
        )

        return model

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            optimizer=self._optimizer,
            loss="categorical_crossentropy",
            metrics=get_minimal_multiclass_metrics()
        )

    def build_training_sequence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 128,
        validation_data: Tuple = None,
        random_state: int = 42
    ) -> Tuple[MixedSequence, MixedSequence]:
        """Return .

        Parameters
        --------------------
        X_train: np.ndarray,
            Node indices for training.
        y_train: np.ndarray,
            Labels to predict.
        batch_size: int = 128,
            Batch size for the sequence.
        validation_data: Tuple = None,
            Tuple containing:
            - Node indices for validation
            - Labels to predict
            If None, no validation data are used.
        random_state: int = 42,
            Random seed to reproduce.

        Returns
        --------------------
        Training and validation MixedSequence
        """
        if (X_train < 0).any():
            raise ValueError(
                "There cannot be negative node indices in the training nodes."
            )
        if (X_train >= self._graph.get_nodes_number()).any():
            raise ValueError(
                "There cannot be node indices in the training nodes higher "
                "than the number of nodes in the graph."
            )
        if (np.bincount(X_train, minlength=self._graph.get_nodes_number()) > 1).any():
            raise ValueError(
                "There cannot be duplicated node indices in the training nodes."
            )
        train_sequence = MixedSequence(
            x=NodeLabelNeighboursSequence(
                self._graph, X_train,
                batch_size=batch_size,
                random_state=random_state,
                support_mirror_strategy=self._support_mirror_strategy
            ),
            y=VectorSequence(
                y_train,
                batch_size=batch_size,
                random_state=random_state
            )
        )
        if validation_data is not None:
            X_validation, y_validation = validation_data
            if (X_validation < 0).any():
                raise ValueError(
                    "There cannot be negative node indices in the validation nodes."
                )
            if (X_validation >= self._graph.get_nodes_number()).any():
                raise ValueError(
                    "There cannot be node indices in the validation nodes higher "
                    "than the number of nodes in the graph."
                )
            if (np.bincount(X_validation, minlength=self._graph.get_nodes_number()) > 1).any():
                raise ValueError(
                    "There cannot be duplicated node indices in the testing nodes."
                )
            if np.isin(X_train, X_validation, assume_unique=True).any():
                raise ValueError(
                    "Train and validation node indices cannot overlap!."
                )
            validation_sequence = MixedSequence(
                x=NodeLabelNeighboursSequence(
                    self._graph, X_validation,
                    batch_size=batch_size,
                    random_state=random_state,
                    support_mirror_strategy=self._support_mirror_strategy
                ),
                y=VectorSequence(
                    y_validation,
                    batch_size=batch_size,
                    random_state=random_state
                )
            )
        else:
            validation_sequence = None
        return train_sequence, validation_sequence

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 128,
        epochs: int = 10000,
        validation_data: Tuple = None,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0,
        early_stopping_patience: int = 100,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0,
        reduce_lr_patience: int = 20,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 1,
        random_state: int = 42,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        X_train: np.ndarray,
            Node IDs reserved for the training.
        y_train: np.ndarray,
            One-hot encoded categorical classes.
        epochs: int = 10000,
            Epochs to train the model for.
        validation_data: Tuple = None,
            Data reserved for validation.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_min_delta: float = 0.1,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 5,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 1,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int = 3,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
        reduce_lr_mode: str = "min",
            Direction of the variation of the monitored metric for learning rate.
        reduce_lr_factor: float = 0.9,
            Factor for reduction of learning rate.
        verbose: int = 1,
            Wethever to show the loading bar.
            Specifically, the options are:
            * 0 or False: No loading bar.
            * 1 or True: Showing only the loading bar for the epochs.
            * 2: Showing loading bar for both epochs and batches.
        **kwargs: Dict,
            Additional kwargs to pass to the Keras fit call.

        Returns
        -----------------------
        Dataframe with training history.
        """
        train_sequence, validation_sequence = self.build_training_sequence(
            X_train, y_train,
            batch_size=batch_size,
            validation_data=validation_data,
            random_state=random_state
        )
        return super().fit(
            train_sequence,
            epochs=epochs,
            validation_data=validation_sequence,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            early_stopping_mode=early_stopping_mode,
            reduce_lr_monitor=reduce_lr_monitor,
            reduce_lr_min_delta=reduce_lr_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_mode=reduce_lr_mode,
            reduce_lr_factor=reduce_lr_factor,
            verbose=verbose,
            **kwargs
        )

    @property
    def embedding(self) -> np.ndarray:
        """Return model embeddings.

        Raises
        -------------------
        NotImplementedError,
            If the current embedding model does not have an embedding layer.
        """
        # We need to drop the first column (feature) of the embedding
        # curresponding to the indices 0, as this value is reserved for the
        # masked values. The masked values are the values used to fill
        # the batches of the neigbours of the nodes.
        return Embedder.embedding.fget(self)[1:]  # pylint: disable=no-member

    def get_embedding_dataframe(self) -> pd.DataFrame:
        """Return terms embedding using given index names."""
        return super().get_embedding_dataframe(self._graph.get_node_names())
