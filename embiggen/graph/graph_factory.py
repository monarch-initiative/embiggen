from typing import Dict
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from numba import typed, types  # type: ignore
from .graph import Graph


class GraphFactory:

    def __init__(
        self,
        default_weight: int = 1,
        default_directed: bool = False,
        default_node_type: str = 'biolink:NamedThing',
        default_edge_type: str = 'biolink:Association',
        **kwargs: Dict
    ):
        """Create new GraphFactory object.

        This object has the task of handling the creation of Graph objects,
        handling all the mess that is unifying the various types of CSVs.

        DO NOT add paths to the Graph class: all paths must be handles here.

        Parameters
        ----------------------------
        default_weight: int = 1,
            The default weight for the node when no weight column is given
        default_directed: bool = False,
            Wethever a edge is directed or not when no column for direction is
            given. By default, the edges are considered not directed.
        default_node_type: str = 'biolink:NamedThing',
            The default type for the nodes when no node type column is given.
        default_edge_type: str = 'biolink:Association',
            The default type for the edges when no edge type column is given.
        **kwargs: Dict
            The kwargs to pass directly to the constructor of the Graph.

        Returns
        -----------------------------
        Return new GraphFactory.
        """
        self._default_directed = default_directed
        self._default_weight = default_weight
        self._default_node_type = default_node_type
        self._default_edge_type = default_edge_type
        self._kwargs = kwargs

    def read_csv(
        self,
        edge_path: str,
        node_path: str = None,
        edge_sep: str = "\t",
        node_sep: str = "\t",
        edge_file_has_header: bool = True,
        node_file_has_header: bool = True,
        start_nodes_column: str = "subject",
        end_nodes_column: str = "object",
        node_types_column: str = "category",
        edge_types_column: str = "edge_label",
        directed_column: str = "is_directed",
        weights_column: str = "weight",
        nodes_columns: str = "id",
        **kwargs: Dict
    ):
        """Return new instance of graph based on given files.

        Parameters
        -----------------------
        edge_path: str,
            Path to the edges file.
        node_path: str = None,
            Path to the nodes file.
        edge_sep: str = "\t",
            Separator to use for the edges file.
        node_sep: str = "\t",
            Separator to use for the nodes file.
        edge_file_has_header: bool = True,
            Whetever to edge files has a header or not.
        node_file_has_header: bool = True,
            Whetever to node files has a header or not.
        start_nodes_column: str = "subject",
            Column to use for the starting nodes. When no header is available,
            use the numeric index curresponding to the column.
        end_nodes_column: str = "object",
            Column to use for the ending nodes. When no header is available,
            use the numeric index curresponding to the column.
        node_types_column: str = "category",
            Column to use for the nodes type. When no header is available,
            use the numeric index curresponding to the column.
        directed_column: str = "is_directed",
            Column to use for the directionality of each edge. When no header is
            avaliable, it assumes non-directed.
        weights_column: str = "weight",
            Column to use for the edges weight. When no header is available,
            use the numeric index curresponding to the column.
        nodes_columns: str = "id",
            Column to use to load the node names, when the node_path argument
            is provided. Parameter is ignored otherwise.
        **kwargs: Dict,
            Additional keyword arguments to pass to the instantiation of a new
            graph object.

        Returns
        ----------------------
        """
        edges_df = pd.read_csv(
            edge_path,
            sep=edge_sep,
            header=([0] if edge_file_has_header else None),
            low_memory=False
        )

        # Dropping duplicated edges
        edges_df = edges_df.drop_duplicates([
            start_nodes_column, end_nodes_column
        ])

        edges = edges_df[[start_nodes_column,
                          end_nodes_column]].values.astype(str)

        numba_nodes = typed.List.empty_list(types.string)

        if node_path is not None:
            nodes_df = pd.read_csv(
                node_path,
                sep=node_sep,
                header=([0] if node_file_has_header else None),
                low_memory=False
            )
            nodes = nodes_df[nodes_columns].values.astype(str)
            for node in nodes:
                numba_nodes.append(node)
        else:
            for node in np.unique(edges):
                numba_nodes.append(node)

        # TODO! Add an exception for when there are more nodes in the edges than in the nodes.

        numba_edges = typed.List.empty_list(types.UniTuple(types.string, 2))
        for start, end in edges:
            numba_edges.append((start, end))

        # Since numba is going to discontinue the support to the reflected list
        # we are going to convert the weights list into a numba list.
        weights = (
            # If provided, we use the list from the dataframe.
            edges_df[weights_column].fillna(self._default_weight).values.tolist()
            # Otherwise if the column is not available.
            if weights_column in edges_df.columns
            # We use the default weight.
            else [self._default_weight]*len(numba_edges)
        )

        numba_weights = typed.List.empty_list(types.float64)
        for weight in weights:
            numba_weights.append(weight)

        # Similarly to what done for the weights, we have to resolve the very
        # same issue also for the

        directed_edges = (
            # If provided, we use the list from the dataframe.
            edges_df[directed_column].fillna(self._default_directed).values.tolist()
            # Otherwise if the column is not available.
            if directed_column in edges_df.columns
            # We use the default weight.
            else [self._default_directed]*len(numba_edges)
        )

        numba_directed = typed.List.empty_list(types.boolean)
        for directed_edge in directed_edges:
            numba_directed.append(directed_edge)

        # We need to convert the node types to a list that numba compatible.

        node_types = (
            # If provided, we use the list from the dataframe.
            nodes_df[node_types_column].fillna(self._default_node_type).values.tolist()
            # Otherwise if the column is not available.
            if (
                node_path is not None and
                node_types_column in nodes_df.columns
            )
            # We use the default weight.
            else [self._default_node_type]*len(numba_nodes)
        )

        unique_node_types = {
            node_type: i
            for i, node_type in enumerate(np.unique(node_types).tolist())
        }

        numba_node_types = typed.List.empty_list(types.int64)
        for node_type in node_types:
            numba_node_types.append(unique_node_types[node_type])

        # We need to convert the edge types to a list that is numba compatible.

        edge_types = (
            # If provided, we use the list from the dataframe.
            nodes_df[edge_types_column].fillna(self._default_edge_type).values.tolist()
            # Otherwise if the column is not available.
            if (
                node_path is not None and
                edge_types_column in nodes_df.columns
            )
            # We use the default weight.
            else [self._default_edge_type]*len(numba_edges)
        )



        unique_edge_types = {
            node_type: i
            for i, node_type in enumerate(np.unique(edge_types).tolist())
        }

        numba_edge_types = typed.List.empty_list(types.int64)
        for node_type in edge_types:
            numba_edge_types.append(unique_edge_types[node_type])

        return Graph(
            edges=numba_edges,
            weights=numba_weights,
            nodes=numba_nodes,
            node_types=numba_node_types,
            edge_types=numba_edge_types,
            directed=numba_directed,
            **kwargs
        )
