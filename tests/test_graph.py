from embiggen.graph import Graph, GraphFactory
from unittest import TestCase
import pytest
from tqdm.auto import tqdm
import numpy as np


class TestGraph(TestCase):

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

        self._factory = GraphFactory()
        self._directed_factory = GraphFactory(default_directed=True)

    def test_setup_from_dataframe(self):
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(path)
                subgraph = graph._graph
                for i in range(len(subgraph._nodes_alias)):
                    assert subgraph.is_node_trap(i) or subgraph.extract_random_node_neighbour(
                        i) in subgraph._nodes_alias[i][0]
                for edge in range(len(subgraph._edges_alias)):
                    assert subgraph.is_edge_trap(edge) or subgraph.extract_random_edge_neighbour(
                        edge) in subgraph._edges_alias[edge][0]

                assert graph.consistent_hash() == graph.consistent_hash()

    def test_unrastered_graph(self):
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(path, random_walk_preprocessing=False)
                with pytest.raises(ValueError):
                    graph.random_walk(1, 1)
                
                assert graph.consistent_hash() == graph.consistent_hash()

    def test_legacy(self):
        """Testing that the normalization process actually works."""
        for path in tqdm(
            self._legacy_paths,
            desc="Testing on legacy"
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(
                    path,
                    edge_file_has_header=False,
                    start_nodes_column=0,
                    end_nodes_column=1,
                    weights_column=2
                )
                subgraph = graph._graph
                for i in range(len(subgraph._nodes_alias)):
                    assert subgraph.is_node_trap(i) or subgraph.extract_random_node_neighbour(
                        i) in subgraph._nodes_alias[i][0]
                for edge in range(len(subgraph._edges_alias)):
                    assert subgraph.is_edge_trap(edge) or subgraph.extract_random_edge_neighbour(
                        edge) in subgraph._edges_alias[edge][0]

    def test_setup_from_custom_dataframe(self):
        graph = self._factory.read_csv(
            "tests/data/small_9606.protein.actions.txt",
            start_nodes_column="item_id_a",
            end_nodes_column="item_id_b",
            weights_column="score"
        )
        graph.random_walk(10, 5)

    def test_random_walk(self):
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy",
            disable=False
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(
                    path, return_weight=10, explore_weight=10)
                all_walks = graph.random_walk(10, 5).numpy()
                subgraph = graph._graph
                assert all(
                    edge in subgraph._edges
                    for walks in all_walks
                    for walk in walks
                    for edge in zip(walk[:-1], walk[1:])
                )
                assert all_walks.shape[0] == subgraph.nodes_number
                assert all(
                    walks.shape[0] == 10
                    for walks in all_walks
                )
                if subgraph.has_traps:
                    assert all(
                        1 <= len(walk) <= 5
                        for walks in all_walks
                        for walk in walks
                    )
                else:
                    assert all(
                        len(walk) == 5
                        for walks in all_walks
                        for walk in walks
                    )

    def test_random_walk_on_legacy(self):
        for path in tqdm(
            self._legacy_paths,
            desc="Testing on non-legacy",
            disable=False
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(
                    path,
                    edge_file_has_header=False,
                    start_nodes_column=0,
                    end_nodes_column=1,
                    weights_column=2,
                    return_weight=10,
                    explore_weight=10
                )
                all_walks = graph.random_walk(10, 5)
                subgraph = graph._graph
                assert all_walks.shape[0] ==subgraph.nodes_number
                assert all(
                    walks.shape[0] == 10
                    for walks in all_walks
                )
                if subgraph.has_traps:
                    assert all(
                        1 <= walk.shape[0] <= 5
                        for walks in all_walks
                        for walk in walks
                    )
                else:
                    assert all(
                        walk.shape[0] == 5
                        for walks in all_walks
                        for walk in walks
                    )

    def test_alias_shape(self):
        for path in tqdm(
            self._paths,
            desc="Testing on non-legacy",
            disable=False
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(
                    path, return_weight=10, explore_weight=10)
                subgraph = graph._graph
                assert all(
                    len(neighbors) == len(j) == len(q) and (
                        subgraph.has_traps or len(q) > 0)
                    for (neighbors, j, q) in subgraph._nodes_alias
                )

    def test_alias_shape_on_legacy(self):
        for path in tqdm(
            self._legacy_paths,
            desc="Testing on non-legacy",
            disable=False
        ):
            for factory in (self._factory, self._directed_factory):
                graph = factory.read_csv(
                    path,
                    edge_file_has_header=False,
                    start_nodes_column=0,
                    end_nodes_column=1,
                    weights_column=2,
                    return_weight=10,
                    explore_weight=10
                )
                subgraph = graph._graph
                assert all(
                    len(neighbors) == len(j) == len(q) and (
                        subgraph.has_traps or len(q) > 0)
                    for (neighbors, j, q) in subgraph._nodes_alias
                )
