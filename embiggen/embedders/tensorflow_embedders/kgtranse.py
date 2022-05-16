"""TransE model."""
from typing import Union, Dict
from ensmallen import Graph
import pandas as pd
import numpy as np
from tensorflow.keras import Model
from .siamese import Siamese


class TransETensorFlow(Siamese):
    """TransE model."""

    def _build_output(
        self,
        *args
    ):
        """Returns the five input tensors, unchanged."""
        return args[:-1]

    @staticmethod
    def model_name() -> str:
        """Returns name of the current model."""
        return "KGTransE"

    @staticmethod
    def requires_node_types() -> bool:
        return True

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
                    ("node_type_embedding", graph.get_unique_node_type_names(), graph.has_unknown_node_types())
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
                ("node_type_embedding", graph.has_unknown_node_types())
            )
        }