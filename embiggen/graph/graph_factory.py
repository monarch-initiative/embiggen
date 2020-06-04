from typing import Dict
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from .graph import Graph
from .csv_utils import check_consistent_lines


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

        Raises
        ----------------------
        ValueError,
            If the provided edges file have malformed lines.
        ValueError,
            If the provided nodes file have malformed lines.
        ValueError,
            If the nodes in the nodes file are duplicated.
        ValueError,
            If the edges file contains node that do not appear in the nodes file.

        Returns
        ----------------------
        New instance of Graph
        """

        if not check_consistent_lines(edge_path, edge_sep):
            raise ValueError(
                "Provided edges file has malformed lines. "
                "The provided lines have different numbers "
                "of the given separator"
            )

        if not (node_path is None or check_consistent_lines(node_path, node_sep)):
            raise ValueError(
                "Provided nodes file has malformed lines. "
                "The provided lines have different numbers "
                "of the given separator"
            )

        tmp_edges_df = pd.read_csv(
            edge_path,
            sep=edge_sep,
            header=(0 if edge_file_has_header else None),
            nrows=1
        )
        edges_df = pd.read_csv(
            edge_path,
            sep=edge_sep,
            usecols=[
                column
                for column in (
                    start_nodes_column,
                    end_nodes_column,
                    edge_types_column,
                    weights_column,
                    directed_column
                )
                if column is not None and column in tmp_edges_df.columns
            ],
            dtype={
                start_nodes_column: "string",
                end_nodes_column: "string",
                edge_types_column: "string",
                weights_column: np.float32,
                directed_column: np.bool
            },
            header=(0 if edge_file_has_header else None)
        )

        # Dropping duplicated edges
        edges_df = edges_df.drop_duplicates([
            start_nodes_column, end_nodes_column
        ])

        edges = edges_df[[
            start_nodes_column, end_nodes_column
        ]].values.astype(str)

        unique_nodes = np.unique(edges)

        if node_path is not None:
            tmp_nodes_df = pd.read_csv(
                node_path,
                sep=node_sep,
                header=(0 if node_file_has_header else None),
                nrows=1
            )
            nodes_df = pd.read_csv(
                node_path,
                sep=node_sep,
                usecols=[
                    column
                    for column in (
                        nodes_columns,
                        node_types_column
                    )
                    if column is not None and column in tmp_nodes_df.columns
                ],
                dtype={
                    nodes_columns: "string",
                    node_types_column: "string"
                },
                header=(0 if node_file_has_header else None)
            )
            nodes = nodes_df[nodes_columns].values.astype(str)

            # Checking if the nodes from edges are contained in nodes file.
            # We create a set since hashing has a O(1) access time.
            nodes_set = set(nodes)
            for node in unique_nodes:
                if node not in nodes_set:
                    raise ValueError((
                        "Edge node {} does not appear "
                        "in the given nodes set."
                    ).format(
                        node
                    ))

            if len(nodes_set) != len(nodes):
                raise ValueError((
                    "There are {} duplicate nodes "
                    "in the given nodes file"
                ).format(len(nodes) - len(nodes_set)))

        else:
            nodes = unique_nodes
        
        #######################################
        # Handling edge weights               #
        #######################################

        if weights_column in edges_df.columns:
            edges_df[weights_column].fillna(
                value=self._default_weight,
                inplace=True
            )
            weights = edges_df[weights_column].values
        else:
            weights = np.full(len(edges), self._default_weight, dtype=np.float32)

        #######################################
        # Handling edge directions            #
        #######################################

        # If provided, we use the list from the dataframe.
        if directed_column in edges_df.columns:
            edges_df[directed_column].fillna(
                value=self._default_directed,
                inplace=True
            )
            directed_edges = edges_df[directed_column].values
        else:
            # Otherwise if the column is not available.
            directed_edges = np.full(len(edges), self._default_directed, dtype=np.bool)

        #######################################
        # Handling node types                 #
        #######################################

        if node_path is not None and node_types_column in nodes_df.columns:
            # If provided, we use the list from the dataframe.
            nodes_df[node_types_column].fillna(
                value=self._default_node_type,
                inplace=True
            )
            node_types = nodes_df[node_types_column].values.astype(str)
        else:
            # Otherwise if the column is not available.
            node_types = np.full(len(nodes), self._default_node_type, dtype=str)

        unique_node_types = {
            node_type: np.int16(i)
            for i, node_type in enumerate(np.unique(node_types))
        }

        numba_node_types = np.empty(len(node_types), dtype=np.int16)
        for i, node_type in enumerate(node_types):
            numba_node_types[i] = unique_node_types[node_type]

        #######################################
        # Handling edge types                 #
        #######################################

        # If provided, we use the list from the dataframe.
        if edge_types_column in edges_df.columns:
            edges_df[edge_types_column].fillna(
                value=self._default_edge_type,
                inplace=True
            )
            edge_types = edges_df[edge_types_column].values
        else:
            # Otherwise if the column is not available.
            edge_types = np.full(len(edges), self._default_edge_type, dtype=str)

        unique_edge_types = {
            edge_type: i
            for i, edge_type in enumerate(np.unique(edge_types))
        }

        numba_edge_types = np.empty(len(edge_types), dtype=np.int64)
        for i, edge_type in enumerate(edge_types):
            numba_edge_types[i] = unique_edge_types[edge_type]

        return Graph(
            edges=edges,
            weights=weights,
            nodes=nodes,
            node_types=numba_node_types,
            edge_types=numba_edge_types,
            directed=directed_edges,
            **kwargs
        )
