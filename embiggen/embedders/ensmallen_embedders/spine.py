"""Module providing abstract Node2Vec implementation."""
from typing import Optional,  Dict, Any, Union
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from ...utils import AbstractEmbeddingModel


class SPINE(AbstractEmbeddingModel):
    """Class implementing the SPINE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        dtype: Optional[str] = "u8"
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        dtype: Optional[str] = "u8"
            Dtype to use for the embedding. Note that an improper dtype may cause overflows.
        """
        self._dtype = dtype
        self._model = models.SPINE(embedding_size=embedding_size)

        super().__init__(
            embedding_size=embedding_size,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                dtype=self._dtype
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
            dtype=self._dtype,
            verbose=verbose,
        ).T
        if return_dataframe:
            return pd.DataFrame(
                node_embedding,
                index=graph.get_nodes_number()
            )
        return node_embedding


    def model_name(self) -> str:
        """Returns name of the model."""
        return "SPINE"