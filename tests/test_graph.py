from embiggen.graph import GraphFactory
from unittest import TestCase
import pytest
from tqdm.auto import tqdm


class TestGraph(TestCase):

    def setUp(self):
        self._test_cases = {
            "non_legacy": {
                "paths": [
                    'tests/data/unweighted_small_graph.txt',
                    'tests/data/small_het_graph_edges.tsv',
                    'tests/data/small_graph.txt',
                ],
                "arguments":[{}]
            },
            "legacy": {
                "paths": [
                    "tests/data/small_graph_LEGACY.txt",
                    "tests/data/small_g2d_test.txt",
                    "tests/data/rand_100nodes_5000edges.graph",
                    "tests/data/ppt_train.txt",
                    "tests/data/ppt_test.txt",
                    *[
                        f"tests/data/ppismall_with_validation/{filename}"
                        for filename in (
                            "neg_test_edges_max_comp_graph",
                            "neg_train_edges_max_comp_graph",
                            "neg_validation_edges_max_comp_graph",
                            "pos_test_edges_max_comp_graph",
                            "pos_train_edges_max_comp_graph",
                            "pos_validation_edges_max_comp_graph"
                        )
                    ],
                    *[
                        f"tests/data/ppismall/{filename}"
                        for filename in (
                            "neg_test_edges",
                            "neg_train_edges",
                            "pos_test_edges",
                            "pos_train_edges",
                        )
                    ],
                    *[
                        f"tests/data/karate/{filename}"
                        for filename in (
                            "neg_test_edges",
                            "neg_train_edges",
                            "neg_validation_edges",
                            "pos_test_edges",
                            "pos_train_edges",
                            "pos_validation_edges",
                        )
                    ]
                ],
                "arguments":[{
                    "start_nodes_column": 0,
                    "end_nodes_column": 1,
                    "weights_column": 2,
                    "edge_file_has_header": False
                }]
            }
        }

        self._factories= [
            GraphFactory(directed=True),
            GraphFactory(directed=False),
            GraphFactory(preprocess=True),
            GraphFactory(preprocess=False)
        ]
        self._verbose= False

    def test_singleton_nodes(self):
        """
            The goal of this test is to see if the factory is able to handle
            graphs containing singleton nodes.
        """
        GraphFactory().read_csv(
            "tests/data/singleton_edges.tsv",
            "tests/data/singleton_nodes.tsv"
        )

    def test_duplicated_nodes(self):
        """
            The goal of this test is to see if the factory raises a proper
            exception when duplicated nodes are present.
        """
        with pytest.raises(ValueError):
            GraphFactory().read_csv(
                "tests/data/singleton_edges.tsv",
                "tests/data/duplicated_nodes.tsv"
            )

    def test_everything_graph(self):
        for factory in tqdm(self._factories, desc="Executing factories", leave=False):
            for test, kwargs in self._test_cases.items():
                for path in tqdm(kwargs["paths"], desc="Executing test on {}".format(test), leave=False):
                    for args in kwargs["arguments"]:
                        graph = factory.read_csv(path, **args)
                        if graph.preprocessed:
                            graph.random_walk(5, 10)
                        else:
                            with pytest.raises(ValueError):
                                graph.random_walk(5, 10)