"""TransE model."""
from typing import Union, Dict
from ensmallen import Graph
import pandas as pd
import numpy as np
from tensorflow.keras import Model
from .siamese import Siamese


class TransETensorFlow(Siamese):
    """TransE model."""

    def __init__(
        self,
        node_embedding_size: int = 100,
        edge_type_embedding_size: int = 100,
        relu_bias: float = 1.0,
        epochs: int = 10,
        batch_size: int = 2**10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 1,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 1,
        use_mirrored_strategy: bool = False,
        optimizer: str = "sgd",
    ):
        """Create new sequence Siamese model.

        Parameters
        -------------------------------------------
        node_embedding_size: int = 100
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        edge_type_embedding_size: int = 100
            Dimension of the embedding for the edge types.
        relu_bias: float = 1.0
            The bias to use for the ReLu.
        epochs: int = 10
            Number of epochs to train the model for.
        batch_size: int = 2**14
            Batch size to use during the training.
        early_stopping_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int = 1
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int = 1
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        optimizer: str = "sgd"
            The optimizer to be used during the training of the model.
        """
        super().__init__(
            node_embedding_size=node_embedding_size,
            edge_type_embedding_size=edge_type_embedding_size,
            relu_bias=relu_bias,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            learning_rate_plateau_min_delta=learning_rate_plateau_min_delta,
            learning_rate_plateau_patience=learning_rate_plateau_patience,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            use_mirrored_strategy=use_mirrored_strategy,
        )

    def _build_output(
        self,
        *args
    ):
        """Returns the five input tensors, unchanged."""
        return args[:-1]

    @staticmethod
    def model_name() -> str:
        """Returns name of the current model."""
        return "TransE"

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return True

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Model,
        return_dataframe: bool
    ) -> Union[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Returns embedding from the model.

        Parameters
        ------------------
        graph: Graph
            The graph that was embedded.
        model: Model
            The Keras model used to embed the graph.
        return_dataframe: bool
            Whether to return a dataframe of a numpy array.
        """
        if return_dataframe:
            return {
                layer_name: pd.DataFrame(
                    self.get_layer_weights(
                        layer_name,
                        model,
                        drop_first_row=drop_first_row
                    ),
                    index=names
                )
                for layer_name, names, drop_first_row in (
                    ("node_embedding", graph.get_node_names(), False),
                    ("edge_type_embedding", graph.get_unique_edge_type_names(), graph.has_unknown_edge_types())
                )
            }
        return {
            layer_name: self.get_layer_weights(
                layer_name,
                model,
                drop_first_row=drop_first_row
            )
            for layer_name, drop_first_row in (
                ("node_embedding", False),
                ("edge_type_embedding", graph.has_unknown_edge_types())
            )
        }