"""Unit test class for testing corner cases in edge-label GCN model."""
from unittest import TestCase

import numpy as np
from dict_hash import sha256
from ensmallen import Graph
from tqdm import tqdm

from embiggen.edge_prediction import GCNEdgePrediction
from embiggen.embedders import HyperSketching
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
from .cached_tests import cache_or_store


class TestGCNEdgePrediction(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        graph = Graph.generate_random_connected_graph(
            node_type="hallo", edge_type="huhu", weight=56.0, number_of_nodes=20
        ) | Graph.generate_random_connected_graph(
            node_type="hallo2",
            edge_type="huhu2",
            weight=56.0,
            number_of_nodes=20,
            minimum_node_id=10,
        )
        self.graph = graph.remove_parallel_edges().add_selfloops("huhu2", 3.0)

    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        if not GCNEdgePrediction.is_available():
            return

        if cache_or_store(
            [
                "./embiggen/embedders/ensmallen_embedders/hyper_sketching.py",
                "./embiggen/edge_prediction/edge_prediction_tensorflow/gcn.py",
                "./embiggen/utils/abstract_gcn.py",
                "./embiggen/utils/abstract_edge_gcn.py",
            ]
        ):
            return

        parametrizations = []
        features = []

        edge_features = HyperSketching()
        edge_features.fit(self.graph)

        rnd = np.random.RandomState(42)

        for use_node_embedding in (True, False):
            for use_node_type_embedding in (True, False):
                for residual_convolutional_layers in (True, False):
                    for kernels in (
                        "Left Normalized Laplacian",
                        ["Left Normalized Laplacian", "Right Normalized Laplacian"],
                        None,
                    ):
                        for use_edge_type_embedding in (True, False):
                            for use_edge_metrics in (True, False):
                                for node_type_features in [
                                    rnd.random(
                                        (self.graph.get_number_of_node_types(), 5)
                                    ),
                                    [
                                        rnd.random(
                                            (self.graph.get_number_of_node_types(), 3)
                                        ),
                                        rnd.random(
                                            (self.graph.get_number_of_node_types(), 7)
                                        ),
                                    ],
                                    None,
                                ]:
                                    for node_features in [
                                        rnd.random(
                                            (self.graph.get_number_of_nodes(), 5)
                                        ),
                                        [
                                            rnd.random(
                                                (self.graph.get_number_of_nodes(), 13)
                                            ),
                                            rnd.random(
                                                (self.graph.get_number_of_nodes(), 17)
                                            ),
                                        ],
                                        None,
                                    ]:
                                        for edge_type_features in [
                                            rnd.random(
                                                (
                                                    self.graph.get_number_of_edge_types(),
                                                    15,
                                                )
                                            ),
                                            [
                                                rnd.random(
                                                    (
                                                        self.graph.get_number_of_edge_types(),
                                                        9,
                                                    )
                                                ),
                                                rnd.random(
                                                    (
                                                        self.graph.get_number_of_edge_types(),
                                                        11,
                                                    )
                                                ),
                                            ],
                                            None,
                                        ]:
                                            for edge_features in [edge_features, None]:
                                                for siamese_node_feature_module in (
                                                    True,
                                                    False,
                                                ):

                                                    if (
                                                        not use_node_embedding
                                                        and not use_node_type_embedding
                                                        and not use_edge_type_embedding
                                                        and not use_edge_metrics
                                                        and node_type_features is None
                                                        and node_features is None
                                                        and edge_features is None
                                                        and edge_type_features is None
                                                    ):
                                                        continue
                                                    features.append(
                                                        dict(
                                                            node_type_features=node_type_features,
                                                            node_features=node_features,
                                                            edge_type_features=edge_type_features,
                                                            edge_features=edge_features,
                                                        )
                                                    )
                                                    parametrizations.append(
                                                        dict(
                                                            use_node_embedding=use_node_embedding,
                                                            use_node_type_embedding=use_node_type_embedding,
                                                            residual_convolutional_layers=residual_convolutional_layers,
                                                            siamese_node_feature_module=siamese_node_feature_module,
                                                            kernels=kernels,
                                                            use_edge_type_embedding=use_edge_type_embedding,
                                                            use_edge_metrics=use_edge_metrics,
                                                        )
                                                    )
        for i, (parametrization, feature_set) in tqdm(
            enumerate(zip(parametrizations, features)),
            total=len(parametrizations),
            desc="Testing parametrization combo",
        ):
            if (
                not parametrization["use_node_embedding"]
                and not parametrization["use_node_type_embedding"]
                and feature_set["node_type_features"] is None
                and feature_set["node_features"] is None
            ):
                parametrization["kernels"] = None
                parametrization["edge_embedding_methods"] = None

            if parametrization["kernels"] is None:
                parametrization["residual_convolutional_layers"] = False

            salt = sha256(
                dict(parametrization=parametrization, feature_set=feature_set),
                use_approximation=True,
            )

            if cache_or_store(
                [
                    "./embiggen/embedders/ensmallen_embedders/hyper_sketching.py",
                    "./embiggen/edge_prediction/edge_prediction_tensorflow/gcn.py",
                    "./embiggen/utils/abstract_gcn.py",
                    "./embiggen/utils/abstract_edge_gcn.py",
                ],
                salt=salt,
            ):
                continue

            model: GCNEdgePrediction = GCNEdgePrediction(
                epochs=1,
                number_of_batches_per_epoch=1,
                number_of_units_per_graph_convolution_layers=4,
                number_of_units_per_ffnn_body_layer=4,
                number_of_units_per_ffnn_head_layer=4,
                number_of_graph_convolution_layers=2
                if parametrization["kernels"] is not None
                else 0,
                node_embedding_size=5,
                node_type_embedding_size=5,
                edge_type_embedding_size=5,
                **parametrization
            )
            try:
                model.fit(graph=self.graph, **feature_set)
                model.predict_proba(graph=self.graph, **feature_set)
                model.predict(graph=self.graph, **feature_set)
            except Exception as e:
                raise ValueError(
                    f"Error raised for parametrization {parametrization} and feature set {feature_set} "
                    f"at the test number {i} out of {len(parametrizations)}."
                ) from e
