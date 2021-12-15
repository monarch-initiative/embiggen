"""Class for creating a Feed-Forward Neural Network model for edge prediction tasks."""
from typing import Union, List, Optional
from ensmallen import Graph
from tensorflow.keras import regularizers  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Dense, Layer  # pylint: disable=import-error,no-name-in-module

import numpy as np
import pandas as pd

from .edge_prediction_model import EdgePredictionModel


class FeedForwardNeuralNetwork(EdgePredictionModel):

    def __init__(
        self,
        graph: Graph,
        units: Optional[List[int]] = None,
        embedding_size: int = None,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        edge_embedding_method: str = "Concatenate",
        optimizer: str = "nadam",
        trainable_embedding: bool = False,
        use_dropout: bool = True,
        dropout_rate: float = 0.5,
        use_edge_metrics: bool = False,
        task_name: str = "EDGE_PREDICTION"
    ):
        """Create new Feed Forward Neural Network object.

        Parameters
        --------------------
        graph: Graph,
            The graph object to base the model on.
        units: Optional[List[int]] = None
            Units to use to build the hidden layers of the model.
        embedding_size: int = None
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        embedding: Union[np.ndarray, pd.DataFrame] = None
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        edge_embedding_method: str = "Concatenate",
            Method to use to create the edge embedding.
        optimizer: str = "nadam",
            Optimizer to use during the training.
        trainable_embedding: bool = False,
            Whether to allow for trainable embedding.
        use_dropout: bool = True,
            Whether to use dropout.
        dropout_rate: float = 0.5,
            Dropout rate.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        task_name: str = "EDGE_PREDICTION",
            The name of the task to build the model for.
            The currently supported task names are `EDGE_PREDICTION` and `EDGE_LABEL_PREDICTION`.
            The default task name is `EDGE_PREDICTION`.
        """
        self._units = [] if units is None else units
        super().__init__(
            graph,
            embedding_size=embedding_size,
            embedding=embedding,
            edge_embedding_method=edge_embedding_method,
            optimizer=optimizer,
            trainable_embedding=trainable_embedding,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            use_edge_metrics=use_edge_metrics,
            task_name=task_name
        )

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for edge prediction."""
        for unit in self._units:
            input_layer = Dense(
                units=unit,
                activation="relu",
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(input_layer)
        if self._task_name == "EDGE_PREDICTION":
            return Dense(
                units=1,
                activation="sigmoid",
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(input_layer)
        elif self._task_name == "EDGE_LABEL_PREDICTION":
            return Dense(
                units=self._graph.get_edge_types_number(),
                activation="softmax",
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(input_layer)
        else:
            ValueError("Unreacheable.")