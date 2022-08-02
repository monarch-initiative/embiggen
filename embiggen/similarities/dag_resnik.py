from typing import List, Optional, Union, Dict
import pandas as pd
import numpy as np
from ensmallen import models, Graph


class DAGResnik:

    def __init__(self, verbose: bool = True):
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
        second_node_ids: List[int]
    ) -> np.ndarray:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_ids: List[int]
            The first node for which to compute the similarity.
        second_node_ids: List[int]
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_ids(first_node_ids, second_node_ids)

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
        second_node_names: List[str]
    ) -> np.ndarray:
        """Return the similarity between the two provided nodes.

        Arguments
        --------------------
        first_node_names: List[str]
            The first node for which to compute the similarity.
        second_node_names: List[str]
            The second node for which to compute the similarity.
        """
        return self._model.get_similarity_from_node_names(first_node_names, second_node_names)

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
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run similarities on.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        similarities = self._model.get_similarities_from_graph(
            graph,
        )

        if return_similarities_dataframe:
            similarities = pd.DataFrame(
                {
                    "similarities": similarities,
                    "sources": graph.get_directed_source_node_ids(),
                    "destinations": graph.get_directed_destination_node_ids(),
                },
            )

        return similarities

    def get_similarities_from_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        source_node_ids: List[int]
            The source nodes of the bipartite graph.
        destination_node_ids: List[int]
            The destination nodes of the bipartite graph.
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
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        source_node_names: List[str]
            The source nodes of the bipartite graph.
        destination_node_names: List[str]
            The destination nodes of the bipartite graph.
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
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_bipartite_graph_from_edge_node_prefixes(
        self,
        graph: Graph,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        source_node_prefixes: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_prefixes: List[str]
            The destination node prefixes of the bipartite graph.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_bipartite_graph_from_edge_node_prefixes(
                source_node_prefixes=source_node_prefixes,
                destination_node_prefixes=destination_node_prefixes,
                directed=True
            ),
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        source_node_types: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_types: List[str]
            The destination node prefixes of the bipartite graph.
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
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        node_ids: List[int]
            The nodes of the bipartite graph.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_ids(
                node_ids=node_ids,
                directed=True
            ),
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        node_names: List[str]
            The nodes of the bipartite graph.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_names(
                node_names=node_names,
                directed=True
            ),
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_prefixes(
        self,
        graph: Graph,
        node_prefixes: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        node_prefixes: List[str]
            The node prefixes of the bipartite graph.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_prefixes(
                node_prefixes=node_prefixes,
                directed=True
            ),
            return_similarities_dataframe=return_similarities_dataframe
        )

    def get_similarities_from_clique_graph_from_node_types(
        self,
        graph: Graph,
        node_types: List[str],
        return_similarities_dataframe: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Execute similarities probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_frequencies: Optional[np.ndarray]
            Optional vector of node frequencies.
        node_types: List[str]
            The node prefixes of the bipartite graph.
        return_similarities_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the similarities is returned as it weights much less.
        """
        return self.get_similarities_from_graph(
            graph.build_clique_graph_from_node_types(
                node_types=node_types,
                directed=True
            ),
            return_similarities_dataframe=return_similarities_dataframe
        )
