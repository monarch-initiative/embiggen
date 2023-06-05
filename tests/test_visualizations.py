"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
import pytest
import numpy as np
import pandas as pd
from embiggen import GraphVisualizer
from ensmallen.datasets.kgobo import CIO, MIAPA
from ensmallen.datasets.networkrepository import Usair97
from embiggen.embedders.ensmallen_embedders.deepwalk_glove import DeepWalkGloVeEnsmallen


class TestGraphVisualizer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        pass

    def test_graph_visualization(self):
        """Test graph visualization."""
        for gr in (CIO() | MIAPA(), Usair97(), MIAPA()):
            node_type_predictions = None
            gr = gr.remove_parallel_edges()
            if gr.has_node_types():
                gr, _ = gr.get_node_label_holdout_graphs(
                    train_size=0.8,
                )
                train_nodes, test_nodes = gr.get_node_label_holdout_indices(
                    train_size=0.7
                )
                node_type_predictions = np.random.randint(
                    gr.get_number_of_node_types(),
                    size=gr.get_number_of_nodes()
                )
            if gr.has_edge_types():
                gr, _ = gr.get_edge_label_holdout_graphs(
                    train_size=0.8,
                )
            for graph in (gr, gr.to_directed()):
                for method in ("PCA", "TSNE", ):
                    decomposition_kwargs = None
                    if method == "TSNE":
                        decomposition_kwargs = dict(n_iter=250)
                    visualization = GraphVisualizer(
                        graph,
                        decomposition_method=method,
                        decomposition_kwargs=decomposition_kwargs,
                        source_node_types_names=graph.get_unique_node_type_names() if graph.has_node_types() else None,
                        destination_node_types_names=graph.get_unique_node_type_names() if graph.has_node_types() else None,
                        source_edge_types_names=graph.get_unique_edge_type_names() if graph.has_edge_types() else None,
                        destination_edge_types_names=graph.get_unique_edge_type_names() if graph.has_edge_types() else None,
                        edge_type_names=graph.get_unique_edge_type_names() if graph.has_edge_types() else None,
                        number_of_subsampled_nodes=40,
                        number_of_subsampled_edges=40,
                        number_of_subsampled_negative_edges=40
                    )
                    for callback_method in (
                        "plot_connected_components",
                        "plot_node_ontologies",
                        "plot_connected_components",
                        "plot_node_degrees",
                        "plot_edge_segments",
                        "plot_nodes",
                        "plot_node_types",
                        "plot_edge_types",
                        "plot_edge_weights",
                        "plot_edges",
                        "plot_positive_and_negative_edges",
                        "plot_positive_and_negative_edges_adamic_adar"
                    ):
                        with pytest.raises(ValueError):
                            visualization.__getattribute__(callback_method)()
                    visualization.fit_and_plot_all("Degree-based SPINE", embedding_size=5)
                    visualization.plot_dot()
                    visualization.plot_edges()
                    visualization.plot_node_degree_distribution()
                    visualization.plot_positive_and_negative_edges()
                    visualization.plot_nodes(annotate_nodes=True, show_edges=True)
                    visualization.plot_node_degrees(
                        annotate_nodes=True,
                        show_edges=True
                    )
                    if not graph.is_directed():
                        visualization.plot_connected_components(
                            annotate_nodes=True,
                            show_edges=True
                        )
                    if graph.has_node_ontologies():
                        visualization.plot_node_ontologies(
                            annotate_nodes=True,
                            show_edges=True
                        )
                    if graph.has_node_types():
                        visualization.plot_node_types(
                            annotate_nodes=True,
                            show_edges=True,
                            train_indices=train_nodes,
                            test_indices=test_nodes,
                            node_type_predictions=node_type_predictions
                        )
                    else:
                        with pytest.raises(ValueError):
                            visualization.plot_node_types(
                                annotate_nodes=True,
                                show_edges=True
                            )
                    if graph.has_edge_types():
                        visualization.plot_edge_types()
                    else:
                        with pytest.raises(ValueError):
                            visualization.plot_edge_types()
                    if graph.has_edge_weights():
                        visualization.plot_edge_weights()
                        visualization.plot_edge_weight_distribution()
                    else:
                        with pytest.raises(ValueError):
                            visualization.plot_edge_weight_distribution()
                        with pytest.raises(ValueError):
                            visualization.plot_edge_weights()
                    visualization.fit_and_plot_all("Degree-based SPINE", embedding_size=2)
                    try:
                        visualization = GraphVisualizer(
                            graph,
                            decomposition_method=method,
                            n_components=3,
                            decomposition_kwargs=decomposition_kwargs,
                        )
                        visualization.fit_and_plot_all(
                            DeepWalkGloVeEnsmallen().into_smoke_test()
                        )
                        visualization.plot_dot()
                        visualization.plot_edges()
                        visualization.plot_nodes(annotate_nodes=True, show_edges=True)
                        visualization.fit_and_plot_all(
                            "Degree-based SPINE",
                            embedding_size=3,
                            include_distribution_plots=False
                        )
                        visualization = GraphVisualizer(
                            graph,
                            decomposition_method=method,
                            n_components=2,
                            rotate=True,
                            fps=2,
                            duration=1,
                            decomposition_kwargs=decomposition_kwargs,
                        )
                        visualization.fit_negative_and_positive_edges("Degree-based SPINE", embedding_size=3)
                        visualization.fit_nodes(pd.DataFrame(
                            np.random.uniform(size=(
                                graph.get_number_of_nodes(),
                                768
                            )).astype(np.float16),
                            index=graph.get_node_names()
                        ))
                        visualization.plot_node_degrees()
                        if graph.has_node_ontologies():
                            visualization.plot_node_types()
                        if not graph.is_directed():
                            visualization.plot_connected_components()
                        visualization.plot_positive_and_negative_edges()
                        if graph.has_node_types():
                            visualization.plot_node_types(
                                train_indices=train_nodes,
                                test_indices=test_nodes,
                                node_type_predictions=node_type_predictions
                            )
                        else:
                            with pytest.raises(ValueError):
                                visualization.plot_node_types()
                        if graph.has_edge_types():
                            visualization.plot_edge_types()
                        else:
                            with pytest.raises(ValueError):
                                visualization.plot_edge_types()
                        visualization.plot_edges()
                        visualization.plot_nodes(annotate_nodes=True)
                    except NameError:
                        print("ddd_subplot might have some problems")
                        pass

