from typing import Dict
import pandas as pd
import numpy as np
from numba import typed, types
from .graph import Graph


class GraphFactory:

    def __init__(
        self,
        default_weight: int = 1,
        default_directed: bool = False,
        default_node_type: str = 'biolink:NamedThing',
        **kwargs: Dict
    ):
        self._default_directed = default_directed
        self._default_weight = default_weight
        self._default_node_type = default_node_type
        self._kwargs = kwargs

    def read_csv(
        self,
        edge_path: str,
        node_path: str = None,
        edge_sep: str = "\t",
        node_sep: str = "\t",
        edge_has_header: bool = True,
        start_nodes_column: str = "subject",
        end_nodes_column: str = "object",
        nodes_type_column: str = "category",
        weights_column: str = "weight",
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
        edge_has_header: bool = True,
            Whetever to edge files has a header or not.
        start_nodes_column: str = "subject",
            Column to use for the starting nodes. When no header is available,
            use the numeric index curresponding to the column.
        end_nodes_column: str = "object",
            Column to use for the ending nodes. When no header is available,
            use the numeric index curresponding to the column.
        nodes_type_column: str = "category",
            Column to use for the nodes type. When no header is available,
            use the numeric index curresponding to the column.
        weights_column: str = "weight",
            Column to use for the edges weight. When no header is available,
            use the numeric index curresponding to the column.
        **kwargs: Dict,
            Additional keyword arguments to pass to the instantiation of a new
            graph object.

        Returns
        ----------------------
        """
        graph_df = pd.read_csv(
            edge_path,
            sep=edge_sep,
            header=([0] if edge_has_header else None)
        )

        # Dropping duplicated edges
        graph_df = graph_df.drop_duplicates(
            [start_nodes_column, end_nodes_column])

        edges = graph_df[[start_nodes_column,
                          end_nodes_column]].values.astype(str)
        numba_edges = typed.List.empty_list(types.UniTuple(types.string, 2))

        for start, end in edges:
            numba_edges.append((start, end))

        numba_nodes = typed.List.empty_list(types.string)
        for (start, end) in edges:
            if start not in numba_nodes:
                numba_nodes.append(start)
            if end not in numba_nodes:
                numba_nodes.append(end)

        # Since numba is going to discontinue the support to the reflected list
        # we are going to convert the weights list into a numba list.
        weights = (
            # If provided, we use the list from the dataframe.
            graph_df[weights_column].values.tolist()
            # Otherwise if the column is not available.
            if weights_column in graph_df.columns
            # We use the default weight.
            else [self._default_weight]*len(numba_edges)
        )

        numba_weights = typed.List.empty_list(types.float64)
        for weight in weights:
            numba_weights.append(weight)

        # Similarly to what done for the weights, we have to resolve the very
        # same issue also for the
        numba_directed = typed.List.empty_list(types.boolean)
        for _ in range(len(numba_edges)):
            numba_directed.append(self._default_directed)

        # Yet again we need to convert the node types to a list that is types
        # and numba compatible.

        nodes_type = (
            # If provided, we use the list from the dataframe.
            graph_df[nodes_type_column].values.tolist()
            # Otherwise if the column is not available.
            if nodes_type_column in graph_df.columns
            # We use the default weight.
            else [self._default_node_type]*len(numba_edges)
        )

        unique_nodes_type = np.unique(nodes_type).tolist()

        numba_nodes_type = typed.List.empty_list(types.int64)
        for node_type in nodes_type:
            numba_nodes_type.append(unique_nodes_type.index(node_type))

        return Graph(
            edges=numba_edges,
            weights=numba_weights,
            nodes=numba_nodes,
            nodes_type=numba_nodes_type,
            directed=numba_directed,
            ** kwargs
        )
