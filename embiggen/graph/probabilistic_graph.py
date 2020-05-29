from multiprocessing import Pool, cpu_count
from typing import Dict

import numpy as np
from numba import njit

from tqdm.auto import tqdm
from typing import Tuple

from .graph import Graph
from .probabilistic_graph_utils import alias_draw, alias_setup


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
        workers: int = -1
            Number of processes to use. Use a number lower or equal to 0 to
            use all available processes. Default is -1.
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
        workers = workers if workers > 0 else cpu_count()

        with Pool(min(workers, self.nodes_number)) as pool:
            self._neighbours_nodes_alias = list(tqdm(
                pool.imap(
                    alias_setup,
                    self._neighbours_weights
                ),
                total=self.nodes_number,
                disable=not verbose,
                desc="Computing node neighbours aliases"
            ))

            self._neighbours_edges_alias = list(tqdm(
                pool.imap(
                    self._weighted_alias_setup,
                    self._edges.keys()
                ),
                total=self.edges_number,
                disable=not verbose,
                desc="Computing edge neighbours aliases"
            ))

            pool.close()
            pool.join()

    def _weighted_alias_setup(self, edge: Tuple[int, int]) -> np.ndarray:
        """Return weighted probabilities for given edge.

        Parameters
        -----------------
        edge: Tuple[int, int],
            Edge for which to compute the probabilities.

        Returns
        -----------------
        Weighted probabilities
        """
        src, dst = edge
        
        probs = np.fromiter((
                self._neighbours_weights[dst][index]
                if self.has_edge((neighbour, src)) else
                self._neighbours_weights[dst][index] / self._p
                if neighbour == src else
                self._neighbours_weights[dst][index] / self._q
                for index, neighbour in enumerate(self._neighbours[dst])
            ),
            dtype=np.int64
        )

        return alias_setup(probs/probs.sum())

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
        return self._neighbours[node][alias_draw(*self._neighbours_nodes_alias[node])]

    def extract_random_edge_neighbour(self, edge: int) -> int:
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
        _, dst = edge
        return self._neighbours[dst][
            alias_draw(*self._neighbours_edges_alias[self._edges[edge]])
        ]
