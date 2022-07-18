"""Unit test class for GraphTransformer objects."""
from unittest import TestCase

import pytest
import numpy as np
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
        for gr in (CIO(), Usair97(), MIAPA()):
            node_type_predictions = None
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
            for graph in (gr, gr.to_directed()):
                for method in ("PCA", "TSNE", "UMAP"):
                    decomposition_kwargs = None
                    if method == "TSNE":
                        decomposition_kwargs = dict(n_iter=1)
                    visualization = GraphVisualizer(
                        graph,
                        decomposition_method=method,
                        decomposition_kwargs=decomposition_kwargs,
                        number_of_subsampled_nodes=20,
                        number_of_subsampled_edges=20,
                        number_of_subsampled_negative_edges=20
                    )
                    visualization.fit_and_plot_all("SPINE", embedding_size=5)
                    visualization.plot_dot()
                    visualization.plot_edges()
                    visualization.plot_node_degree_distribution()
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
                        visualization.plot_edge_weight_distribution()
                    else:
                        with pytest.raises(ValueError):
                            visualization.plot_edge_weight_distribution()
                    visualization.fit_and_plot_all("SPINE", embedding_size=2)
                    visualization = GraphVisualizer(
                        graph,
                        decomposition_method=method,
                        n_components=3,
                        decomposition_kwargs=decomposition_kwargs,
                    )
                    visualization.fit_and_plot_all("SPINE", embedding_size=5)
                    visualization.fit_and_plot_all(
                        DeepWalkGloVeEnsmallen().into_smoke_test()
                    )
                    visualization.plot_dot()
                    visualization.plot_edges()
                    visualization.plot_nodes(annotate_nodes=True, show_edges=True)
                    visualization.fit_and_plot_all(
                        "SPINE",
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
                    visualization.fit_edges("SPINE", embedding_size=3)
                    visualization.fit_nodes("SPINE", embedding_size=3)
                    visualization.plot_node_degrees()
                    if graph.has_node_ontologies():
                        visualization.plot_node_types()
                    if not graph.is_directed():
                        visualization.plot_connected_components()
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
