from embiggen.graph import GraphFactory
from unittest import TestCase
import pytest
from tqdm.auto import tqdm
from IPython import embed


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
        self._verbose = False

    def test_singleton_nodes(self):
        """
            The goal of this test is to see if the factory is able to handle
            graphs containing singleton nodes.
        """
        graph = self._factory.read_csv(
            "tests/data/singleton_edges.tsv",
            "tests/data/singleton_nodes.tsv"
        )

    def test_setup_from_dataframe(self):
        for path in tqdm(
            self._paths,
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(path)
                subgraph = graph._graph
                self.assertTrue(all([
                    dst1 == dst2 and 0 <= dst1 < len(subgraph._nodes_neighboring_edges)
                    for dst1, (_, dst2) in zip(subgraph._destinations, subgraph._edges)
                ]))
                self.assertTrue(all([
                    0 <= edge_id < len(subgraph._edges)
                    for _, edge_id in subgraph._edges.items()
                ]))
                self.assertTrue(all(
                    0 <= edge_id < len(subgraph._edges)
                    for edges in subgraph._nodes_neighboring_edges
                    for edge_id in edges
                ))
                for edge in range(len(subgraph._edges_alias)):
                    self.assertTrue(
                        subgraph.is_edge_trap(edge) or
                        subgraph.extract_random_edge_neighbour(edge)[1]
                        in subgraph._nodes_neighboring_edges[
                            subgraph._destinations[edge]
                        ]
                    )

                self.assertEqual(graph.consistent_hash(),
                                 graph.consistent_hash())

    def test_unrastered_graph(self):
        for path in tqdm(
            self._paths,
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(path, random_walk_preprocessing=False)
                with pytest.raises(ValueError):
                    graph.random_walk(1, 1)

                self.assertEqual(graph.consistent_hash(),
                                 graph.consistent_hash())

    def test_legacy(self):
        """Testing that the normalization process actually works."""
        for path in tqdm(
            self._legacy_paths,
            disable=not self._verbose,
            desc="Testing on legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(
                    path,
                    edge_file_has_header=False,
                    start_nodes_column=0,
                    end_nodes_column=1,
                    weights_column=2
                )
                subgraph = graph._graph
            
                for _, edge_id in subgraph._edges.items():
                    self.assertTrue(0 <= edge_id < len(subgraph._edges))

                self.assertTrue(all(
                    0 <= edge_id < len(subgraph._edges)
                    for edges in subgraph._nodes_neighboring_edges
                    for edge_id in edges
                ))
                
                for i in range(len(subgraph._nodes_alias)):
                    self.assertTrue(
                        subgraph.is_node_trap(i) or
                        subgraph.extract_random_node_neighbour(i)[1]
                        in subgraph._nodes_neighboring_edges[i]
                    )
                for edge in range(len(subgraph._edges_alias)):
                    self.assertTrue(
                        subgraph.is_edge_trap(edge) or
                        subgraph.extract_random_edge_neighbour(edge)[1]
                        in subgraph._nodes_neighboring_edges[
                            subgraph._destinations[edge]
                        ]
                    )

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
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(
                    path, return_weight=10, explore_weight=10)
                walks_number = 10
                walks_length = 5
                walks = graph.random_walk(walks_number, walks_length).numpy()
                subgraph = graph._graph
                self.assertTrue(all(
                    edge in subgraph._edges
                    for walk in walks
                    for edge in zip(walk[:-1], walk[1:])
                ))
                self.assertEqual(
                    walks.shape[0], subgraph.nodes_number*walks_number)
                if subgraph.has_traps:
                    self.assertTrue(all(
                        1 <= len(walk) <= walks_length
                        for walk in walks
                    ))
                else:
                    self.assertTrue(all(
                        len(walk) == walks_length
                        for walk in walks
                    ))

    def test_random_walk_on_legacy(self):
        for path in tqdm(
            self._legacy_paths,
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(
                    path,
                    edge_file_has_header=False,
                    start_nodes_column=0,
                    end_nodes_column=1,
                    weights_column=2,
                    return_weight=10,
                    explore_weight=10
                )
                walks_number = 10
                walks_length = 5
                walks = graph.random_walk(walks_number, walks_length)
                subgraph = graph._graph
                self.assertEqual(
                    walks.shape[0], subgraph.nodes_number*walks_number)
                if subgraph.has_traps:
                    self.assertTrue(all(
                        1 <= walk.shape[0] <= walks_length
                        for walk in walks
                    ))
                else:
                    self.assertTrue(all(
                        walk.shape[0] == walks_length
                        for walk in walks
                    ))

    def test_alias_shape(self):
        for path in tqdm(
            self._paths,
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
                graph = factory.read_csv(
                    path, return_weight=10, explore_weight=10)
                subgraph = graph._graph
                self.assertTrue(all(
                    len(j) == len(q) and (
                        subgraph.has_traps or len(q) > 0)
                    for (j, q) in subgraph._nodes_alias
                ))

    def test_alias_shape_on_legacy(self):
        for path in tqdm(
            self._legacy_paths,
            disable=not self._verbose,
            desc="Testing on non-legacy"
        ):
            for factory in (self._factory, ):
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
                self.assertTrue(all(
                    len(j) == len(q) and (
                        subgraph.has_traps or len(q) > 0)
                    for (j, q) in subgraph._nodes_alias
                ))
