"""Module providing abstract Node2Vec implementation."""
from typing import  Dict, Any, Union
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from scipy import rand
from ...utils import AbstractEmbeddingModel


class TransE(AbstractEmbeddingModel):
    """Class implementing the TransE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        renormalize: bool = True,
        random_state: int = 42
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        renormalize: bool = True
            Whether to renormalize at each loop, by default true.
        dtype: int = "u8"
            Dtype to use for the embedding. Note that an improper dtype may cause overflows.
        """
        self._renormalize = renormalize
        self._random_state = random_state
        self._model = models.TransE(
            embedding_size=embedding_size,
            renormalize=renormalize,
            random_state=random_state
        )

        super().__init__(
            embedding_size=embedding_size,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                renormalize = self._renormalize,
                random_state = self._random_state,
            )
        }

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            graph,
            verbose=verbose,
        ).T
        if return_dataframe:
            return pd.DataFrame(
                node_embedding,
                index=graph.get_nodes_number()
            )
        return node_embedding


    def name(self) -> str:
        """Returns name of the model."""
        return "TransE"