"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List

import numpy as np
import pandas as pd
from numba import njit, prange

from .node_transformer import NodeTransformer


@njit(parallel=True)
def numba_hadamard(sources: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Execute hadamard in parallel within numba."""
    results = np.empty_like(sources, dtype=np.float_)
    for i in prange(sources.shape[0]):  # pylint: disable=not-an-iterable
        results[i] = np.multiply(sources[i], destinations[i])
    return results


@njit(parallel=True)
def numba_average(sources: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Execute average in parallel within numba."""
    results = np.empty_like(sources, dtype=np.float_)
    for i in prange(sources.shape[0]): # pylint: disable=not-an-iterable
        results[i] = (sources[i] + destinations[i]) / 2
    return results


@njit(parallel=True)
def numba_weightedL1(sources: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Execute weightedL1 in parallel within numba."""
    results = np.empty_like(sources, dtype=np.float_)
    for i in prange(sources.shape[0]): # pylint: disable=not-an-iterable
        results[i] = np.abs(sources[i] - destinations[i])
    return results


@njit(parallel=True)
def numba_weightedL2(sources: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Execute weightedL2 in parallel within numba."""
    results = np.empty_like(sources, dtype=np.float_)
    for i in prange(sources.shape[0]): # pylint: disable=not-an-iterable
        results[i] = (sources[i], destinations[i])**2
    return results


class EdgeTransformer:
    """EdgeTransformer class to convert edges to edge embeddings."""

    methods = {
        "hadamard": numba_hadamard,
        "average": numba_average,
        "weightedL1": numba_weightedL1,
        "weightedL2": numba_weightedL2
    }

    def __init__(self, method: str = "hadamard"):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'hadamard', 'average', 'weightedL1', 'weightedL2' or
            a custom lambda that receives two numpy arrays with the nodes
            embedding and returns the edge embedding.
        """
        if isinstance(method, str) and method not in EdgeTransformer.methods:
            raise ValueError((
                "Given method '{}' is not supported. "
                "Supported methods are {}, or alternatively a lambda."
            ).format(
                method, ", ".join(list(EdgeTransformer.methods.keys()))
            ))
        self._method = EdgeTransformer.methods[method]
        self._transformer = NodeTransformer()

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
        """
        self._transformer.fit(embedding)

    def transform(self, sources: List[str], destinations: List[str]) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources: List[str],
            List of source nodes whose embedding is to be returned.
        destinations:List[str],
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
            self._transformer.transform(destinations),
        )
