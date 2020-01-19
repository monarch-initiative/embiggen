import unittest
from unittest import TestCase
import os
from n2v import LinkPrediction, CSFGraph


class TestLinkPrediction(TestCase):

    def test_init(self):
        training_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'karate.train'
        )
        test_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'karate.test'
        )

        embedding_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'disease.embedded'
        )

        test_graph = CSFGraph(training_file)
        training_graph = CSFGraph(test_file)

        parameters = {
            'edge_embedding_method': "hadamard",
            'portion_false_edges': 1
        }

        lp = LinkPrediction(
            training_graph,
            test_graph,
            embedding_file,
            params=parameters
        )

        lp.output_diagnostics_to_logfile()
        lp.predict_links()
        lp.output_Logistic_Reg_results()
