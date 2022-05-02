"""Class for creating a MultiLayerPerceptron model for edge prediction tasks."""
from typing import Union
import numpy as np
import pandas as pd
from ensmallen import Graph
from .feed_forward_neural_network import FeedForwardNeuralNetwork


class MultiLayerPerceptron(FeedForwardNeuralNetwork):

    def __init__(
        self,
        graph: Graph,
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
        """Create new MultiLayerPerceptron model object.

        Parameters
        --------------------
        graph: Graph,
            The graph object to base the model on.
        embedding_size: int = None,
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
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
        super().__init__(
            graph,
            units=[32, 32],
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
