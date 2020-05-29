from typing import List, Union, Tuple, Dict
import numpy as np


class Graph:

    def __init__(
        self,
        edges: Union[List[Tuple[str, str]], np.ndarray],
        nodes: Union[List[str], np.ndarray] = None,
        weights: Union[List, np.ndarray, float] = 1,
        edge_types: Union[List[str], str] = 'biolink:Association',
        node_types: Union[List[str], str] = 'biolink:NamedThing',
        normalize_weights: bool = False
    ):
        """Crate a new instance of a undirected graph with given edges.

        Parameters
        ---------------------
        edges: Union[List[Tuple[str, str]], np.ndarray],
            List of edges of the graph.
        nodes: Union[List[str], np.ndarray] = None,
            List of the nodes of the graph. By default, the list is obtained
            from the given list of edges.
        weights: Union[List[float], float] = 1,
            Either the weights for each source and sink or the default weight
            to use. By default, the weight is 1.
        edge_types: Union[List[str], str] = 'biolink:Association',
            Either the edge types for each source and sink or the default edge
            type to use. By default, the edge type is 'biolink:Association'.
        node_types: Union[List[str], str] = 'biolink:NamedThing',
            Either the node types for each source and sink or the default node
            type to use. By default, the node type is 'biolink:NamedThing'.
        normalize_weights: bool = False,
            Wethever to normalize the graph weights per neighbour.
            By default False.

        Raises
        ---------------------
        ValueError,
            If given weights number does not match given edges number.
        ValueError,
            If given constant weight is 0.
        ValueError,
            If given edge types number does not match given edges number.
        ValueError,
            If given node types number does not match given nodes number.

        Returns
        ---------------------
        New instance of graph.
        """
        self._edges_number = len(edges)
        constant_weight = not isinstance(weights, (List, np.ndarray))
        if not constant_weight and len(weights) != self._edges_number:
            raise ValueError(
                "Given weights number does not match given edges number."
            )
        if constant_weight and weights == 0:
            raise ValueError(
                "Given constant weight is zero."
            )
        if isinstance(edge_types, List) and len(edge_types) != self._edges_number:
            raise ValueError(
                "Given edge types number does not match given edge number."
            )

        # Retrieving unique edges if list of nodes was not provided.
        if nodes is None:
            nodes = np.unique(edges)
        self._nodes_number = len(nodes)
        self._nodes_indices = np.arange(self._nodes_number)

        if isinstance(node_types, List) and len(node_types) != self._nodes_number:
            raise ValueError(
                "Given node types number does not match given nodes number."
            )

        # Creating mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        self._nodes = dict(zip(nodes, self._nodes_indices))

        # Creating mapping of edges and integer ID.
        # The map looks like the following:
        # {
        #   (0, 1): 0,
        #   (0, 2): 1
        # }
        self._edges = {}
        # Each node has a list of neighbours.
        # These lists are initialized as empty.
        neighbours = [[] for _ in range(self._nodes_number)]
        # Each node has a list of neighbours weights.
        # These lists are initialized as empty, if a weight
        neighbours_weights = [[] for _ in range(self._nodes_number)]
        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        for i, (start, end) in enumerate(edges):
            start, end = sorted((self._nodes[start], self._nodes[end]))
            self._edges[(start, end)] = i
            # We populate the list of the neighbours for both end of the edge.
            neighbours[start].append(end)
            neighbours[end].append(start)
            # We populate the list of the neighbours's weights for both sides.
            weight = weights if constant_weight else weights[i]
            neighbours_weights[start].append(weight)
            neighbours_weights[end].append(weight)

        # We prepare the generator to transform the lists of neighbours and
        # of weights in numpy arrays. We use generator to avoid iterating
        # multiple times on the same list.
        neighbours_generator = (
            np.array(local_neighbours)
            for local_neighbours in neighbours
        )
        neighbours_weights_generator = (
            np.array(local_neighbours_weights)
            for local_neighbours_weights in neighbours_weights
        )
        # If required, we normalize the neighbours weights, that is we create
        # a new generator with the added feature of normalizing the resulting
        # numpy arrays.
        if normalize_weights:
            neighbours_weights_generator = (
                local_neighbours_weights / local_neighbours_weights.sum()
                for local_neighbours_weights in neighbours_weights_generator
            )
        # Finally now we can iterate over the two generators and save the
        # obtained result into two class private variables.
        self._neighbours = list(neighbours_generator)
        self._neighbours_weights = list(neighbours_weights_generator)

    @property
    def nodes_indices(self) -> int:
        """Return the number of nodes in the graph."""
        return self._nodes_indices

    @property
    def nodes_number(self) -> int:
        """Return the number of nodes in the graph."""
        return self._nodes_number