from embiggen.graph import GraphFactory
from embiggen import RandomWalker
from unittest import TestCase
import pytest
import numpy as np
from tqdm.auto import tqdm


class TestRandomWalker(TestCase):

    def setUp(self):
        self._paths = [
            'tests/data/unweighted_small_graph.txt',
            'tests/data/small_het_graph_edges.tsv',
            'tests/data/small_graph.txt',
        ]
        self._legacy_space_paths = [
            "tests/data/karate.train",
            "tests/data/karate.test",
        ]
        self._legacy_paths = [
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
        ]

        self._verbose = True
        self._factory = GraphFactory()
        self._walker = RandomWalker(verbose=False)

    def test_random_walk(self):
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy",
            disable=not self._verbose
        ):
            graph = self._factory.read_csv(path, return_weight=10, explore_weight=10)
            self._walker.walk(graph, 10, 5)

    def test_random_walk_on_legacy(self):
        for path in tqdm(
            self._legacy_paths,
            desc="Testing on non-legacy",
            disable=not self._verbose
        ):
            graph = self._factory.read_csv(
                path,
                edge_has_header=False,
                start_nodes_column=0,
                end_nodes_column=1,
                weights_column=2,
                return_weight=10,
                explore_weight=10
            )
            self._walker.walk(graph, 10, 5)
