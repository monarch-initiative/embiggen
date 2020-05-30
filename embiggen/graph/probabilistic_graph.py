from typing import Dict, Tuple

import numpy as np
from numba import njit, jit

from tqdm.auto import tqdm

from .graph import Graph
from .probabilistic_graph_utils import alias_draw, alias_setup, new_alias_draw


class ProbabilisticGraph(Graph):

    def __init__(self, p: float, q: float, workers: int = -1, verbose: bool = True, **kwargs: Dict):
        """Create the ProbabilisticGraph object.

        Parameters
        ----------------------
        p: float,
            Return parameter.
            # TODO: better specify the effect, impact and valid ranges of this
            # parameter.
        q: float,
            In-out parameter
            # TODO: better specify the effect, impact and valid ranges of this
            # parameter.
        verbose: bool = True,
            Wethever to show or not the loading bar.
        **kwargs: Dict,
            Parameters to pass to the parent Graph class.

        Raises
        -----------------------
        ValueError,
            If given parameter p is not a strictly positive number.
        ValueError,
            If given parameter q is not a strictly positive number.
        """
        super().__init__(normalize_weights=True, **kwargs)
        if not isinstance(p, (int, float)) or p <= 0:
            raise ValueError(
                "Given parameter p is not a stricly positive number."
            )
        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError(
                "Given parameter q is not a stricly positive number."
            )
        self._p = p
        self._q = q
        self._new_alias_draw = new_alias_draw()

        self._neighbours_nodes_alias = [
            alias_setup(neighbours_weights)
            for neighbours_weights in self._neighbours_weights
        ]

        self._neighbours_edges_alias = [
            {
                neighbour: self._weighted_alias_setup(src, neighbour)
                for neighbour in self._neighbours[src]
            }
            for src in self.nodes_indices
        ]

    def _weighted_alias_setup(self, src: int, dst: int) -> np.ndarray:
        """Return weighted probabilities for given edge.

        Parameters
        -----------------
        edge: Tuple[int, int],
            Edge for which to compute the probabilities.

        Returns
        -----------------
        Weighted probabilities
        """

        total = 0
        neighbours = self._neighbours[dst]
        probs = np.empty(self.nodes_number)

        for index, neighbour in enumerate(neighbours):
            weight = self._neighbours_weights[dst][index]
            if self.has_edge((neighbour, src)):
                pass
            elif neighbour == src:
                weight = weight / self._p
            else:
                weight = weight / self._q
            total += weight
            probs[index] = weight

        return alias_setup(probs/total)

    def extract_random_node_neighbour(self, node: int) -> int:
        """Return a random adiacent node to the one associated to node.
        The Random is extracted by using the normalized weights of the edges
        as probability distribution. 

        Parameters
        ----------
        node: int
            The index of the node that is to be considered.

        Returns
        -------
        The index of a random adiacent node to node.
        """
        return self._neighbours[node][self._new_alias_draw(*self._neighbours_nodes_alias[node])]

    def extract_random_edge_neighbour(self, src: int, dst: int) -> int:
        """Return a random adiacent node to the one associated to node.
        The Random is extracted by using the normalized weights of the edges
        as probability distribution. 

        Parameters
        ----------
        node: int
            The index of the node that is to be considered.

        Returns
        -------
        The index of a random adiacent node to node.
        """
        return self._neighbours[dst][self._new_alias_draw(*self._neighbours_edges_alias[src][dst])]
