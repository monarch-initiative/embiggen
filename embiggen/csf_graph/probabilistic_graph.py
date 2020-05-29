from multiprocessing import cpu_count, Pool
from .graph import Graph
from typing import Dict
from .probabilistic_graph_utils import alias_setup, alias_draw
from numba import njit


class ProbabilisticGraph(Graph):

    def __init__(self, workers: int = -1, **kwargs: Dict):
        """Create the ProbabilisticGraph object.

        Parameters
        ----------------------
        workers: int = -1
            Number of processes to use. Use a number lower or equal to 0 to
            use all available processes. Default is -1.
        **kwargs: Dict,
            Parameters to pass to the parent Graph class.
        """
        super().__init__(normalize_weights=True, **kwargs)
        workers = workers if workers <= 0 else cpu_count()

        with Pool(min(workers, self._neighbours_weights)) as pool:
            self._neighbours_alias = pool.map(
                alias_setup,
                self._neighbours_weights
            )
            pool.close()
            pool.join()

    @njit
    def extract_random_neighbour(self, node_index: int) -> int:
        """Return a random adiacent node to the one associated to node_index.
        The Random is extracted by using the normalized weights of the edges
        as probability distribution. 

        Parameters
        ----------
        node_index:int
            The index of the node that is to be considered.

        Returns
        -------
        The index of a random adiacent node to node_index.
        """
        return alias_draw(self._neighbours_alias[node_index])
