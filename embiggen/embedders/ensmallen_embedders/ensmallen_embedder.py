"""Submodule providing abstract class for embedding graph nodes using Ensmallen."""
from typing import Union, Dict, Optional
from ensmallen import Graph
from ..embedders_utils import detect_graph_node_embedding_oddities
from ...utils import validate_verbose
import pandas as pd
import numpy as np


class EnsmallenEmbedder:
    """Abstract class providing graph node embedders based on Ensmallen."""

    def __init__(
        self,
        embedding_size: int = 100,
        verbose: bool = True
    ):
        """Create new EnsmallenEmbedder object.
        
        Parameters
        --------------------
        embedding_size: int = 100
            Number of dimensions of the node embedding
        verbose: bool = True
            Whether to show loading bars

        Raises
        --------------------
        ValueError
            If the provided embedding size is less than 1.
        """
        if not isinstance(embedding_size, int) or embedding_size < 1:
            raise ValueError(
                (
                    "The embedding size must be a strictly positive integer, "
                    "but you have provided {}."
                ).format(embedding_size)
            )
        validate_verbose(verbose)
        
        self._embedding_size = embedding_size
        self._verbose = verbose

    def _fit_transform(
        self,
        graph: Graph,
    ) -> np.ndarray:
        """Method to be reimplemented in child classes to implement.
        
        Parameters
        -------------------
        graph: Graph
            The ensmallen graph to embed.
        """
        raise NotImplementedError(
            "The method `_fit_transform` must be implemented "
            "in child classes."
        )

    def fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return node embedding for provided graph.

        Parameters
        ---------------------
        graph: Graph
            The ensmallen graph to embed.
        return_dataframe: bool = True
            Whether to return the dataframe or directly
            a numpy array. Do note that using a DataFrame
            increases non trivially the amount of memory
            required for a graph node embedding.
        """
        detect_graph_node_embedding_oddities(graph)
        node_embedding = self._fit_transform(graph)
        if return_dataframe:
            return pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        return node_embedding