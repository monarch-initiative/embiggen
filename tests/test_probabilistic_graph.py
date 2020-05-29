from embiggen.graph import ProbabilisticGraph, GraphFactory
from unittest import TestCase
import pytest
import numpy as np
from tqdm.auto import tqdm


class TestGraph(TestCase):

    def setUp(self):
        self._paths = [
            'tests/data/unweighted_small_graph.txt',
            'tests/data/small_het_graph_edges.tsv',
            'tests/data/small_graph.txt',
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
        self._factory = GraphFactory(ProbabilisticGraph, verbose=False)

    def test_non_legacy(self):
        """Testing that the normalization process actually works."""
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy",
            disable= not self._verbose
        ):
            graph=self._factory.read_csv(path)
            for i in graph.nodes_indices:
                assert graph.extract_random_neighbour(
                    i) in graph._neighbours[i]


    def test_legacy(self):
        """Testing that the normalization process actually works."""
        for path in tqdm(
            self._legacy_paths,
            desc="Testing on legacy",
            disable= not self._verbose
        ):
            graph=self._factory.read_csv(
                path,
                edge_has_header=False,
                start_nodes_column=0,
                end_nodes_column=1,
                weights_column=2
            )
            for i in graph.nodes_indices:
                assert graph.extract_random_neighbour(
                    i) in graph._neighbours[i]

    def test_setup_from_custom_dataframe(self):
        # TODO: integrate all other remaining columns
        graph=self._factory.read_csv(
            "tests/data/small_9606.protein.actions.txt",
            start_nodes_column="item_id_a",
            end_nodes_column="item_id_b",
            weights_column="score"
        )
        for i in graph.nodes_indices:
            assert graph.extract_random_neighbour(
                i) in graph._neighbours[i]
