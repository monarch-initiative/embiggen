from typing import List, Optional, Union, Dict, Tuple
import pandas as pd
import numpy as np
from ensmallen import models, Graph


class DAGResnik:

    def __init__(self, verbose: bool = True):
        """Create new Resnik similarity model."""
        self._model = models.DAGResnik(verbose)

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
        self._model.fit(
            graph,
            node_counts=node_counts,
            node_frequencies=node_frequencies
        )

    def get_similarity_from_node_id(
        self,
        first_node_id: int,
        second_node_id: int
    ) -> float:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_id: int
            The first node for which to compute the similarity.
        second_node_id: int
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_id(first_node_id, second_node_id)

    def get_similarity_from_node_ids(
        self,
        first_node_ids: List[int],
        second_node_ids: List[int],
        minimum_similarity: Optional[float] = 0.0,
    ) -> np.ndarray:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_ids: List[int]
            The first node for which to compute the similarity.
        second_node_ids: List[int]
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_ids(first_node_ids, second_node_ids, minimum_similarity)

    def get_similarity_from_node_name(
        self,
        first_node_name: str,
        second_node_name: str
    ) -> float:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_name: str
            The first node for which to compute the similarity.
        second_node_name: str
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_name(first_node_name, second_node_name)

    def get_similarity_from_node_names(
        self,
        first_node_names: List[str],
        second_node_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
    ) -> np.ndarray:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_names: List[str]
            The first node for which to compute the similarity.
        second_node_names: List[str]
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_names(first_node_names, second_node_names, minimum_similarity)

    def get_pairwise_similarities(
        self,
        graph: Optional[Graph] = None,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities on the provided graph.

        Parameters
        --------------------
        graph: Optional[Graph] = None
            The graph to run similarities on.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        similarities = self._model.get_pairwise_similarities()

        if return_similarities_dataframe and graph is not None:
            similarities = pd.DataFrame(
                similarities,
                columns=graph.get_node_names(),
                index=graph.get_node_names(),
            )

        return similarities

    def get_similarities_from_graph(
        self,
        graph: Graph,
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run similarities on.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        (edge_node_ids, similarities) = self._model.get_similarities_from_graph(
            graph,
            minimum_similarity
        )

        if return_similarities_dataframe:
            if not graph.has_edges():
                similarities = pd.DataFrame({
                    "similarity": [],
                    "source": [],
                    "destination": [],
                })
            else:
                similarities = pd.DataFrame(
                    {
                        "similarity": similarities,
                        "source": edge_node_ids[:, 0],
                        "destination": edge_node_ids[:, 1],
                    },
                )

        return similarities

    def get_similarities_from_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_ids: List[int]
            The source nodes of the bipartite graph.
        destination_node_ids: List[int]
            The destination nodes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_bipartite_graph_from_edge_node_ids(
                source_node_ids=source_node_ids,
                destination_node_ids=destination_node_ids,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_names: List[str]
            The source nodes of the bipartite graph.
        destination_node_names: List[str]
            The destination nodes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_bipartite_graph_from_edge_node_names(
                source_node_names=source_node_names,
                destination_node_names=destination_node_names,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_bipartite_graph_from_edge_node_prefixes(
        self,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, Tuple[List[Tuple[str, str]], np.ndarray]]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        source_node_prefixes: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_prefixes: List[str]
            The destination node prefixes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        edge_node_names, similarities = self._model.get_similarity_from_node_prefixes(
            first_node_prefixes=source_node_prefixes,
            second_node_prefixes=destination_node_prefixes,
            minimum_similarity=minimum_similarity,
        )

        if not return_similarities_dataframe:
            return edge_node_names, similarities
        
        edge_node_names = pd.DataFrame(
            edge_node_names,
            columns=["source", "destination"],
        )

        edge_node_names["similarity"] = similarities
        
        return edge_node_names

    def get_similarities_from_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_types: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_types: List[str]
            The destination node prefixes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_bipartite_graph_from_edge_node_types(
                source_node_types=source_node_types,
                destination_node_types=destination_node_types,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_ids: List[int]
            The nodes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_ids(
                node_ids=node_ids,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_names: List[str]
            The nodes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_names(
                node_names=node_names,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_prefixes(
        self,
        node_prefixes: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, Tuple[List[Tuple[str, str]], np.ndarray]]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        node_prefixes: List[str]
            The node prefixes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_bipartite_graph_from_edge_node_prefixes(
            source_node_prefixes=node_prefixes,
            destination_node_prefixes=node_prefixes,
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )        


    def get_similarities_from_clique_graph_from_node_types(
        self,
        graph: Graph,
        node_types: List[str],
        minimum_similarity: Optional[float] = 0.0,
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_types: List[str]
            The node prefixes of the bipartite graph.
        minimum_similarity: Optional[float] = 0.0
            Minimum similarity to be kept. Values below this amount are filtered.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_types(
                node_types=node_types,
                directed=True
            ),
            minimum_similarity=minimum_similarity,
            return_similarities_dataframe=return_similarities_dataframe
        )
