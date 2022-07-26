import pytest
from unittest import TestCase
from embiggen.edge_label_prediction.edge_label_prediction_tensorflow.graph_sage import GraphSAGEEdgeLabelPrediction
from ensmallen.datasets.string import SpeciesTree
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE

class TestGCNEdgeLabelPrediction(TestCase):

    def setUp(self):
        pass

    def test_all_edge_embedding_methods(self):
        model = GraphSAGEEdgeLabelPrediction(
            epochs=1,
            edge_feature_names=["A", "B", "C"]
        )

        with pytest.raises(RuntimeError):
            model.fit(
                SpeciesTree(),
                node_features=DegreeSPINE(embedding_size=4)
            )