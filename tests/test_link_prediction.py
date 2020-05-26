import os
import unittest
from parameterized import parameterized
from embiggen import LinkPrediction, CSFGraph

class TestLinkPrediction(unittest.TestCase):
    def setUp(self) -> None:

        #file_dir = 'data/ppismall_with_validation/'
        file_dir = os.path.join(os.path.dirname(__file__), 'data/ppismall_with_validation/')

        self.pos_train_graph = CSFGraph(os.path.join(file_dir, 'pos_train_edges_max_comp_graph'))
        self.pos_valid_graph = CSFGraph(os.path.join(file_dir, 'pos_validation_edges_max_comp_graph'))
        self.pos_test_graph = CSFGraph(os.path.join(file_dir, 'pos_test_edges_max_comp_graph'))
        self.neg_train_graph = CSFGraph(os.path.join(file_dir, 'neg_train_edges_max_comp_graph'))
        self.neg_valid_graph = CSFGraph(os.path.join(file_dir, 'neg_validation_edges_max_comp_graph'))
        self.neg_test_graph = CSFGraph(os.path.join(file_dir, 'neg_test_edges_max_comp_graph'))
        self.test_embeddings = os.path.join(file_dir, 'test.embeddings')

    @parameterized.expand([
                            ("LR",),
                            ("RF",),
                            ("MLP",),
                            ("FFNN",),
                            ("MultiModalFFNN",),
    ])
    def test_classifiers(self, classifier_name):
        lp = LinkPrediction(
                            self.pos_train_graph,
                            self.pos_valid_graph,
                            self.pos_test_graph,
                            self.neg_train_graph,
                            self.neg_valid_graph,
                            self.neg_test_graph,
                            self.test_embeddings,
                            'hadamard',
                            classifier=classifier_name,
                            skip_valid=True,
                            output="output"
                            )

        lp.prepare_edge_and_node_labels()
        lp.predict_links()
        lp.get_classifier_results()


if __name__ == '__main__':
    unittest.main()
