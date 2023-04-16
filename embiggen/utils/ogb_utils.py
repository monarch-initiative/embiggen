"""Tools relative to OGB."""
from typing import Dict
from ensmallen import Graph
import numpy as np
from tqdm.auto import tqdm
from sanitize_ml_labels import sanitize_ml_labels
from ogb.linkproppred import LinkPropPredDataset


def convert_ogb_link_prediction_dataset_to_grape(
    dataset_name: str
) -> Dict[str, Graph]:
    """Returns list of graphs from provided dataset name.

    Parameters
    ------------------
    dataset_name: str
        Name of OGB dataset to retrieve.
    """
    dataset = LinkPropPredDataset(name=dataset_name)

    number_of_nodes = dataset.graph["num_nodes"]

    graphs = {}

    for holdout_name, edge_holdouts in tqdm(
        dataset.get_edge_split().items(),
        desc=f"Parsing OGB graph {dataset_name}",
        total=len(dataset.get_edge_split()),
        leave=False,
        dynamic_ncols=True
    ):
        for holdout_type, edges in edge_holdouts.items():
            graph_name: str = sanitize_ml_labels(
                f"{dataset_name} {holdout_name} {holdout_type}")
            file_name: str = "{}.csv".format(
                graph_name.lower().replace(" ", "_"))
            np.savetxt(file_name, edges, delimiter=",", fmt="%i")

            graphs[f"{holdout_name}_{holdout_type}"] = Graph.from_csv(
                directed=False,
                name=graph_name,
                edge_path=file_name,
                sources_column_number=0,
                destinations_column_number=1,
                edge_list_header=False,
                edge_list_numeric_node_ids=True,
                nodes_number=number_of_nodes,
            )

    return graphs
