"""Unit test class for GraphTransformer objects."""
from platform import node
from typing import Callable
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
    
    def generic_test(self, callback):
        for return_similarities_dataframe in (True, False):
            for return_node_names in (False, True):
                if return_node_names and not return_similarities_dataframe:
                    with pytest.raises(NotImplementedError):
                        callback(
                        return_similarities_dataframe,
                        return_node_names
                    )
                else:
                    callback(
                        return_similarities_dataframe,
                        return_node_names
                    )
    
    def test_resnik_node_ids(self):
        self.generic_test(
            lambda x, y: self._model.get_similarities_from_clique_graph_node_ids(
                node_ids=[0, 1],
                minimum_similarity=1,
                return_similarities_dataframe=x,
                return_node_names=y
            )
        )

    def test_resnik_node_names(self):
        self.generic_test(
            lambda x, y: self._model.get_similarities_from_clique_graph_node_names(
                node_names=["HP:0000001", "HP:0000002"],
                minimum_similarity=1,
                return_similarities_dataframe=x,
                return_node_names=y
            )
        )
    
    def test_resnik_node_prefixes(self):
        self.generic_test(
            lambda x, y: self._model.get_similarities_from_clique_graph_node_prefixes(
                node_prefixes=["HP:00000"],
                minimum_similarity=1,
                return_similarities_dataframe=x,
                return_node_names=y
            )
        )
    
    def test_resnik_node_type_ids(self):
        self.generic_test(
            lambda x, y: self._model.get_similarities_from_clique_graph_node_type_ids(
                node_type_ids=[0],
                minimum_similarity=1,
                return_similarities_dataframe=x,
                return_node_names=y
            )
        )
    
    def test_resnik_node_type_names(self):
        self.generic_test(
            lambda x, y: self._model.get_similarities_from_clique_graph_node_type_names(
                node_type_names=["biolink:PhenotypicFeature"],
                minimum_similarity=1,
                return_similarities_dataframe=x,
                return_node_names=y
            )
        )
