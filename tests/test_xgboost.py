"""Module to test the XGBoost models for node-label, edge-label and edge prediction."""
from ensmallen import Graph
from ensmallen.datasets.kgobo import MIAPA
from ensmallen.datasets.linqs import Cora, get_words_data

from embiggen.edge_label_prediction import XGBEdgeLabelPrediction
from embiggen.edge_prediction import XGBEdgePrediction
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.embedders.ensmallen_embedders.hyper_sketching import \
    HyperSketching
from embiggen.node_label_prediction import XGBNodeLabelPrediction

from .cached_tests import cache_or_store


def test_xgb_node_label_prediction_smoke_test():
    """Smoke test for the XGBoost node-label prediction model."""
    if cache_or_store([
        "tests/test_xgboost.py",
        "embiggen/embedders/ensmallen_embedders/degree_spine.py",
        "embiggen/node_label_prediction/node_label_prediction_xgboost/xgboost_node_label_prediction.py",
        "embiggen/node_label_prediction/sklearn_like_node_label_prediction_adapter.py",
        "embiggen/node_label_prediction/node_label_prediction_model.py",
        "embiggen/utils/abstract_models/abstract_classifier_model.py",
    ]):
        return
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
        model = XGBNodeLabelPrediction().into_smoke_test()
        model.fit(g, node_features=data)
        model.predict(g, node_features=data)
        model.predict_proba(g, node_features=data)


def test_xgb_edge_prediction_smoke_test():
    """Smoke test for the XGBoost edge-label prediction model."""
    if cache_or_store([
        "tests/test_xgboost.py",
        "embiggen/embedders/ensmallen_embedders/degree_spine.py",
        "embiggen/edge_prediction/edge_prediction_xgboost/xgboost_edge_prediction.py",
        "embiggen/edge_prediction/sklearn_like_edge_prediction_adapter.py",
        "embiggen/edge_prediction/edge_prediction_model.py",
        "embiggen/utils/abstract_models/abstract_classifier_model.py",
    ]):
        return
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
        model = XGBEdgePrediction().into_smoke_test()
        model.fit(g, node_features=data, edge_features=HyperSketching())
        model.predict(g, node_features=data, edge_features=HyperSketching())
        model.predict_proba(g, node_features=data,
                            edge_features=HyperSketching())


def test_xgb_edge_label_prediction_smoke_test():
    """Smoke test for the XGBoost edge-label prediction model."""
    if cache_or_store([
        "tests/test_xgboost.py",
        "embiggen/embedders/ensmallen_embedders/degree_spine.py",
        "embiggen/edge_label_prediction/edge_label_prediction_xgboost/xgboost_edge_label_prediction.py",
        "embiggen/edge_label_prediction/sklearn_like_edge_label_prediction_adapter.py",
        "embiggen/edge_label_prediction/edge_label_prediction_model.py",
        "embiggen/utils/abstract_models/abstract_classifier_model.py",
    ]):
        return
    graph, data = get_words_data(Cora())
    red = graph.set_all_edge_types("red")
    blue = MIAPA().remove_singleton_nodes().set_all_edge_types("blue")
    binary = red | blue

    for g, data in ((binary, None),):
        if data is None:
            data = DegreeSPINE().into_smoke_test().fit_transform(g)
        model = XGBEdgeLabelPrediction().into_smoke_test()
        model.fit(g, node_features=data, edge_features=HyperSketching())
        model.predict(g, node_features=data, edge_features=HyperSketching())
        model.predict_proba(g, node_features=data,
                            edge_features=HyperSketching())
