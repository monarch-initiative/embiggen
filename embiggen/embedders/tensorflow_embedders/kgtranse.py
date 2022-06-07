"""TransE model."""
from typing import Union, Dict
from ensmallen import Graph
import pandas as pd
import numpy as np
from tensorflow.keras import Model
from embiggen.embedders.tensorflow_embedders.siamese import Siamese
from embiggen.utils.abstract_models import EmbeddingResult

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
    ) -> EmbeddingResult:
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
            result = {
                layer_name: pd.DataFrame(
                    self.get_layer_weights(
                        layer_name,
                        model,
                        drop_first_row=drop_first_row
                    ),
                    index=names
                )
                for layer_name, names, drop_first_row in (
                    ("node_embeddings", graph.get_node_names(), False),
                    ("edge_type_embeddings", graph.get_unique_edge_type_names(), graph.has_unknown_edge_types()),
                    ("node_type_embeddings", graph.get_unique_node_type_names(), graph.has_unknown_node_types()),
                )
            }
        else:
            result = {
                layer_name: self.get_layer_weights(
                    layer_name,
                    model,
                    drop_first_row=drop_first_row
                )
                for layer_name, drop_first_row in (
                    ("node_embeddings", False),
                    ("edge_type_embeddings", graph.has_unknown_edge_types()),
                    ("node_type_embeddings", graph.has_unknown_node_types()),
                )
            }
        
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            **result
        )

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return True

    @staticmethod
    def task_involves_edge_types() -> bool:
        """Returns whether the model task involves edge types."""
        return True

    @staticmethod
    def task_involves_node_types() -> bool:
        """Returns whether the model task involves node types."""
        return True