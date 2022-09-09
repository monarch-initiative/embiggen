"""Unit test class for GraphTransformer objects."""
from platform import node
from tqdm.auto import tqdm
from unittest import TestCase
import pytest
import numpy as np
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.similarities import DAGResnik
from ensmallen.datasets.kgobo import HP


class TestDAGResnik(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph = HP(directed=True)\
            .filter_from_names(
                edge_type_names_to_keep=["biolink:subclass_of"],
                node_prefixes_to_keep=["HP:"]
        )\
            .to_transposed()\
            .remove_disconnected_nodes()
        self._graph.enable()
        self._model = DAGResnik()
        self._model.fit(
            self._graph,
            node_counts={
                node_name: 1
                for node_name in self._graph.get_node_names()
            }
        )

    def test_resnik_api1_a(self):
        self._model.get_similarity_from_node_id(10, 20)

    def test_resnik_api1_b(self):
        self._model.get_similarity_from_node_ids([10], [20])

    def test_resnik_api1_c(self):
        self._model.get_similarity_from_node_name(
            self._graph.get_node_name_from_node_id(0),
            self._graph.get_node_name_from_node_id(1)
        )

    def test_resnik_api1_d(self):
        self._model.get_similarity_from_node_names(
            [self._graph.get_node_name_from_node_id(0)],
            [self._graph.get_node_name_from_node_id(1)]
        )

    def test_resnik_api2(self):
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_graph(
                self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
            )

    def test_resnik_api3(self):
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_bipartite_graph_from_edge_node_ids(
                graph=self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
                source_node_ids=[0, 1, 2, 3],
                destination_node_ids=[4, 5, 6, 7],
            )

    def test_resnik_api4(self):
        first_names = self._graph.get_node_names()[:5]
        second_names = self._graph.get_node_names()[5:10]
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_bipartite_graph_from_edge_node_names(
                graph=self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
                source_node_names=first_names,
                destination_node_names=second_names,
            )

    def test_resnik_api5(self):
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_bipartite_graph_from_edge_node_prefixes(
                graph=self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
                source_node_prefixes=["OIO"],
                destination_node_prefixes=["IAO", "CIO"],
            )

    def test_resnik_api5(self):
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_clique_graph_from_node_ids(
                graph=self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
                node_ids=[0, 1, 2],
            )

    def test_resnik_api6(self):
        first_names = self._graph.get_node_names()[:5]
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_clique_graph_from_node_names(
                graph=self._graph,
                return_similarities_dataframe=return_similarities_dataframe,
                node_names=first_names,
            )

    def test_resnik_api7(self):
        for return_similarities_dataframe in (True, False):
            self._model.get_similarities_from_clique_graph_from_node_prefixes(
                return_similarities_dataframe=return_similarities_dataframe,
                node_prefixes=["HP:000000"],
                minimum_similarity=3
            )
