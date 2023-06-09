"""Module to test the Ridge Classifier models for node-label, edge-label and edge prediction."""
from embiggen.node_label_prediction.node_label_prediction_lightgbm.lgbm_node_label_prediction import (
    LightGBMNodeLabelPrediction,
)
from embiggen.edge_prediction.edge_prediction_lightgbm.lgbm_edge_prediction import (
    LightGBMEdgePredictionModel,
)
from embiggen.edge_label_prediction.edge_label_prediction_lightgbm.lgbm_edge_label_prediction import (
    LightGBMEdgeLabelPrediction,
)
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
from ensmallen.datasets.kgobo import MIAPA
from ensmallen import Graph
from ensmallen.datasets.linqs import Cora, get_words_data
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE


def test_lgbm_node_label_prediction_smoke_test():
    """Smoke test for the Ridge Classifier node-label prediction model."""
    graph, data = get_words_data(Cora())
    graph = graph.remove_singleton_nodes()
    red = graph.set_all_node_types("red")
    binary_graph = (
        red | MIAPA().remove_edge_types().set_all_node_types("blue")
    ).add_selfloops()
    another_binary_graph = Graph.generate_random_connected_graph(
        random_state=100,
        number_of_nodes=100,
        node_type="red",
    ) | Graph.generate_random_connected_graph(
        minimum_node_id=100,
        random_state=1670,
        number_of_nodes=100,
        node_type="blue",
    )

    for g, data in (
        (graph, data),
        (binary_graph, None),
        (another_binary_graph, None),
    ):
        if data is None:
            data = DegreeSPINE().into_smoke_test().fit_transform(g)
        model = LightGBMNodeLabelPrediction().into_smoke_test()
        model.fit(g, node_features=data)
        model.predict(g, node_features=data)
        model.predict_proba(g, node_features=data)


def test_lgbm_edge_prediction_smoke_test():
    """Smoke test for the Ridge Classifier edge-label prediction model."""
    graph, data = get_words_data(Cora())
    graph = graph.remove_singleton_nodes()
    red = graph.set_all_node_types("red")
    multilabel_graph = (
        Cora().remove_edge_weights().remove_edge_types() | red
    ).add_selfloops()
    binary_graph = (
        red | MIAPA().remove_edge_types().set_all_node_types("blue")
    ).add_selfloops()
    another_binary_graph = Graph.generate_random_connected_graph(
        random_state=100,
        number_of_nodes=100,
        node_type="red",
    ) | Graph.generate_random_connected_graph(
        minimum_node_id=100,
        random_state=1670,
        number_of_nodes=100,
        node_type="blue",
    )

    for g, data in (
        (graph, data),
        (multilabel_graph, None),
        (binary_graph, None),
        (another_binary_graph, None),
    ):
        if data is None:
            data = DegreeSPINE().into_smoke_test().fit_transform(g)
        model = LightGBMEdgePredictionModel().into_smoke_test()
        model.fit(g, node_features=data, edge_features=HyperSketching())
        model.predict(g, node_features=data, edge_features=HyperSketching())
        model.predict_proba(g, node_features=data, edge_features=HyperSketching())


def test_lgbm_edge_label_prediction_smoke_test():
    """Smoke test for the Ridge Classifier edge-label prediction model."""
    graph, data = get_words_data(Cora())
    red = graph.set_all_edge_types("red")
    blue = MIAPA().remove_singleton_nodes().set_all_edge_types("blue")
    binary = red | blue

    for g, data in ((binary, None),):
        if data is None:
            data = DegreeSPINE().into_smoke_test().fit_transform(g)
        model = LightGBMEdgeLabelPrediction().into_smoke_test()
        model.fit(g, node_features=data, edge_features=HyperSketching())
        model.predict(g, node_features=data, edge_features=HyperSketching())
        model.predict_proba(g, node_features=data, edge_features=HyperSketching())
