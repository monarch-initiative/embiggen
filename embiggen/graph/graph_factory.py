from typing import Dict, Type
import pandas as pd


class GraphFactory:

    def __init__(self, product_class: Type, **kwargs: Dict):
        self._product_class = product_class
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
        **kwargs: Dict,
            Additional keyword arguments to pass to the instantiation of a new
            graph object.

        Returns
        ----------------------
        """
        edges = pd.read_csv(
            edge_path,
            sep=edge_sep,
            header=([0] if edge_has_header else None)
        )
        return self._product_class(
            edges=edges[[start_nodes_column, end_nodes_column]].values,
            **(
                dict(weights=edges[weights_column].values)
                if weights_column in edges.columns else {}
            ),
            **kwargs,
            **self._kwargs
        )
