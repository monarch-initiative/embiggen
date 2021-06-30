"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List

import numpy as np
import pandas as pd

from .node_transformer import NodeTransformer


class EdgeTransformer:
    """EdgeTransformer class to convert edges to edge embeddings."""

    methods = {
        "Hadamard": lambda x1, x2: np.multiply(x1, x2, out=x1),
        "Sum": lambda x1, x2: np.add(x1, x2, out=x1),
        "Average": lambda x1, x2: np.divide(np.add(x1, x2, out=x1), 2, out=x1),
        "L1": lambda x1, x2: np.subtract(x1, x2, out=x1),
        "AbsoluteL1": lambda x1, x2: np.abs(np.subtract(x1, x2, out=x1), out=x1),
        "L2": lambda x1, x2: np.power(np.subtract(x1, x2, out=x1), 2, out=x1),
        "Concatenate": lambda x1, x2: np.hstack((x1, x2)),
        None: lambda x1, x2: np.vstack((x1, x2)).T,
    }

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_node_mapping: bool = False,
        support_mirrored_strategy: bool = False,
    ):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: str = "Hadamard",
            Method to use for the embedding.
            If None is used, we return instead the numeric tuples.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        """
        if isinstance(method, str) and method not in EdgeTransformer.methods:
            raise ValueError((
                "Given method '{}' is not supported. "
                "Supported methods are {}, or alternatively a lambda."
            ).format(
                method, ", ".join(list(EdgeTransformer.methods.keys()))
            ))
        self._transformer = NodeTransformer(
            numeric_node_ids=method is None,
            aligned_node_mapping=aligned_node_mapping,
            support_mirrored_strategy=support_mirrored_strategy
        )
        self._method_name = method
        self._method = EdgeTransformer.methods[self._method_name]

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._transformer.numeric_node_ids

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._method_name

    def fit(self, embedding: pd.DataFrame):
        """Fit the model.

        Parameters
        -------------------------
        embedding: pd.DataFrame,
            Embedding to use to fit the transformer.
            This is a pandas DataFrame and NOT a numpy array because we need
            to be able to remap correctly the vector embeddings in case of
            graphs that do not respect the same internal node mapping but have
            the same node set. It is possible to remap such graphs using
            Ensmallen's remap method but it may be less intuitive to users.

        Raises
        -------------------------
        ValueError,
            If the given method is None there is no need to call the fit method.
        """
        if self._method is None:
            raise ValueError(
                "There is no need to call the fit when edge method is None."
            )
        self._transformer.fit(embedding)

    def transform(self, sources: List[str], destinations: List[str]) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources: List[str],
            List of source nodes whose embedding is to be returned.
        destinations: List[str],
            List of destination nodes whose embedding is to be returned.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        return self._method(
            self._transformer.transform(sources),
            self._transformer.transform(destinations)
        )
