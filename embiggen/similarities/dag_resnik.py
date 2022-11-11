from typing import List, Optional, Union, Dict, Tuple
import pandas as pd
import numpy as np
from ensmallen import models, Graph


class DAGResnik:

    def __init__(self, verbose: bool = True):
        """Create new Resnik similarity model."""
        self._model = models.DAGResnik(verbose)
        self._graph = None

    def fit(
        self,
        graph: Graph,
        node_counts: Dict[str, float],
        node_frequencies: Optional[np.ndarray] = None,
    ):
        """Fit the Resnik similarity model.

        Parameters
        --------------------
        graph: Graph
            The graph to run similarities on.
        node_counts: Dict[str, float]
            Counts to compute the terms frequencies.
        node_frequencies: Optional[np.ndarray] = None
            Optional vector of node frequencies.
        """
        self._graph = graph
        self._model.fit(
            graph,
            node_counts=node_counts,
            node_frequencies=node_frequencies
        )

    def _normalize_output(
        self,
        edge_node_ids: np.ndarray,
        similarities: np.ndarray,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Normalize output to provided standard.
        
        Parameters
        ---------------------
        edge_node_ids: np.ndarray
            The edge node IDs composing the edges.
        similarities: np.ndarray
            Resnik similarity scores.
        return_similarities_dataframe: bool = False
            Whether to return the data as a DataFrame.
            Do note that this will require more RAM.
        return_node_names: bool = False
            Whether to return the node names.
            Do note that this will require SIGNIFICANTLY more RAM.
        """
        if return_node_names and not return_similarities_dataframe:
            raise NotImplementedError(
                "It is not currently supported to return the node names "
                "when it is not requested to return a pandas DataFrame. "
                "This is not supported as the RAM requirements for common "
                "use cases are large enough to be unfeaseable on most systems."
            )

        if not return_similarities_dataframe:
            return (edge_node_ids, similarities)

        return pd.DataFrame({
            "source": (
                self._graph.get_node_names_from_node_ids(edge_node_ids[:, 0])
                if return_node_names
                else edge_node_ids[:, 0]
            ),
            "destination": (
                self._graph.get_node_names_from_node_ids(edge_node_ids[:, 1])
                if return_node_names
                else edge_node_ids[:, 1]
            ),
            "resnik_score": similarities
        })

    def get_similarities_from_bipartite_graph_node_ids(
        self,
        source_node_ids: List[str],
        destination_node_ids: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the bipartite portion

        Parameters
        --------------------
        source_node_ids: List[int]
            The source node ids defining a bipartite graph.
        destination_node_ids: List[int]
            The destination node ids defining a bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        if isinstance(source_node_ids, int):
            source_node_ids = [source_node_ids]

        if isinstance(destination_node_ids, int):
            destination_node_ids = [destination_node_ids]
        
        return self._normalize_output(
            *self._model.get_node_ids_and_similarity_from_node_ids(
                first_node_ids=source_node_ids,
                second_node_ids=destination_node_ids,
                minimum_similarity=minimum_similarity
            ),
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_bipartite_graph_node_names(
        self,
        source_node_names: List[str],
        destination_node_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the bipartite portion

        Parameters
        --------------------
        source_node_names: List[str]
            The source node names defining a bipartite graph.
        destination_node_names: List[str]
            The destination node names defining a bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        if isinstance(source_node_names, str):
            source_node_names = [source_node_names]

        if isinstance(destination_node_names, str):
            destination_node_names = [destination_node_names]
        
        return self._normalize_output(
            *self._model.get_node_ids_and_similarity_from_node_names(
                first_node_names=source_node_names,
                second_node_names=destination_node_names,
                minimum_similarity=minimum_similarity
            ),
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_bipartite_graph_node_prefixes(
        self,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the bipartite portion

        Parameters
        --------------------
        source_node_prefixes: List[str]
            The source node prefixes defining a bipartite graph.
        destination_node_prefixes: List[str]
            The destination node prefixes defining a bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        if isinstance(source_node_prefixes, str):
            source_node_prefixes = [source_node_prefixes]

        if isinstance(destination_node_prefixes, str):
            destination_node_prefixes = [destination_node_prefixes]
        
        return self._normalize_output(
            *self._model.get_node_ids_and_similarity_from_node_prefixes(
                first_node_prefixes=source_node_prefixes,
                second_node_prefixes=destination_node_prefixes,
                minimum_similarity=minimum_similarity
            ),
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_bipartite_graph_node_type_ids(
        self,
        source_node_type_ids: List[int],
        destination_node_type_ids: List[int],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the bipartite portion

        Parameters
        --------------------
        source_node_type_ids: List[int]
            The source node type ids defining a bipartite graph.
        destination_node_type_ids: List[int]
            The destination node type ids defining a bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        if isinstance(source_node_type_ids, int):
            source_node_type_ids = [source_node_type_ids]

        if isinstance(destination_node_type_ids, int):
            destination_node_type_ids = [destination_node_type_ids]
        
        return self._normalize_output(
            *self._model.get_node_ids_and_similarity_from_node_type_ids(
                first_node_type_ids=source_node_type_ids,
                second_node_type_ids=destination_node_type_ids,
                minimum_similarity=minimum_similarity
            ),
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_bipartite_graph_node_type_names(
        self,
        source_node_type_names: List[str],
        destination_node_type_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        source_node_type_names: List[str]
            The source node type names defining a bipartite graph.
        destination_node_type_names: List[str]
            The destination node type names defining a bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        if isinstance(source_node_type_names, str):
            source_node_type_names = [source_node_type_names]

        if isinstance(destination_node_type_names, str):
            destination_node_type_names = [destination_node_type_names]

        return self._normalize_output(
            *self._model.get_node_ids_and_similarity_from_node_type_names(
                first_node_type_names=source_node_type_names,
                second_node_type_names=destination_node_type_names,
                minimum_similarity=minimum_similarity
            ),
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_clique_graph_node_ids(
        self,
        node_ids: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the clique portion

        Parameters
        --------------------
        node_ids: List[int]
            The node type ids defining a clique.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        return self.get_similarities_from_bipartite_graph_node_ids(
            source_node_ids=node_ids,
            destination_node_ids=node_ids,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_clique_graph_node_names(
        self,
        node_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the clique portion

        Parameters
        --------------------
        node_names: List[str]
            The node type ids defining a clique.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        return self.get_similarities_from_bipartite_graph_node_names(
            source_node_names=node_names,
            destination_node_names=node_names,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_clique_graph_node_prefixes(
        self,
        node_prefixes: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the clique portion

        Parameters
        --------------------
        node_prefixes: List[str]
            The node type ids defining a clique.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        return self.get_similarities_from_bipartite_graph_node_prefixes(
            source_node_prefixes=node_prefixes,
            destination_node_prefixes=node_prefixes,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_clique_graph_node_type_ids(
        self,
        node_type_ids: List[int],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the clique portion

        Parameters
        --------------------
        node_type_ids: List[int]
            The node type ids defining a clique.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        return self.get_similarities_from_bipartite_graph_node_type_ids(
            source_node_type_ids=node_type_ids,
            destination_node_type_ids=node_type_ids,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )

    def get_similarities_from_clique_graph_node_type_names(
        self,
        node_type_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False,
        return_node_names: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Execute similarities probabilities on the provided graph clique portion.

        Parameters
        --------------------
        node_type_names: List[str]
            The node type names defining a clique.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as requires much less RAM.
        return_node_names: bool = False
            Whether to return the node names or node IDs associated to the scores.
            By default we return the node ids, which require much less memory.
        """
        return self.get_similarities_from_bipartite_graph_node_type_names(
            source_node_type_names=node_type_names,
            destination_node_type_names=node_type_names,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe,
            return_node_names=return_node_names
        )
