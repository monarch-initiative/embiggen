import multiprocessing as mp
from .graph import Graph
from .probabilistic_graph_utils import alias_setup, alias_draw

class ProbabilisticGraph(Graph):

    def __init__(self, workers:int = -1, **kwargs):
        super().__init__(normalize_weights=True, **kwargs)

        with mp.Pool(mp.cpu_count()) as pool:
            self._neighbours_alias = pool.map(
                alias_setup,
                self._neighbours_weights
            )
            pool.close()
            pool.join()

    def extract_random_neighbour(self, node_index : int) -> int:
        """Extract a random adiacent node to the one associated to node_index.
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
