from multiprocessing import cpu_count, Pool
from .graph import Graph
from typing import Dict
from .probabilistic_graph_utils import alias_setup, alias_draw
from numba import njit
from tqdm.auto import tqdm


class ProbabilisticGraph(Graph):

    def __init__(self, workers: int = -1, verbose: bool = True, **kwargs: Dict):
        """Create the ProbabilisticGraph object.

        Parameters
        ----------------------
        workers: int = -1
            Number of processes to use. Use a number lower or equal to 0 to
            use all available processes. Default is -1.
        verbose: bool = True,
            Wethever to show or not the loading bar.
        **kwargs: Dict,
            Parameters to pass to the parent Graph class.
        """
        super().__init__(normalize_weights=True, **kwargs)
        workers = workers if workers > 0 else cpu_count()

        with Pool(min(workers, self.nodes_number)) as pool:
            self._neighbours_alias = list(tqdm(
                pool.imap(
                    alias_setup,
                    self._neighbours_weights
                ),
                total=self.nodes_number,
                disable=not verbose,
                desc="Computing neighbours aliases"
            ))
            pool.close()
            pool.join()

    def extract_random_neighbour(self, node: int) -> int:
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
        return self._neighbours[node][alias_draw(*self._neighbours_alias[node])]
