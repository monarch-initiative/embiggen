
import timeit
from IPython import embed
import numpy as np
from tqdm.auto import tqdm
import silence_tensorflow.auto
from embiggen import GraphFactory, Graph


def test_function(path='tests/data/small_het_graph_edges.tsv'):
    factory = GraphFactory(default_directed=False)

    graph = factory.read_csv(path, return_weight=10, explore_weight=10)

    walks = graph.random_walk(20, 5)
    if (np.abs(walks) > 1000).any():
        print("CRISTO SIGNORE REDENTORE PORCO DIO")
    else:
        print("Thanks")
    return walks

class TestGraph():

    def __init__(self):
        self._paths = [
            'tests/data/unweighted_small_graph.txt',
            'tests/data/small_het_graph_edges.tsv',
            'tests/data/small_graph.txt',
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
                print(subgraph._edges.get((0, 1, 0), -1337))
                for i in range(len(subgraph._nodes_alias)):
                    try:
                        subgraph.is_node_trap(i) or subgraph.extract_random_node_neighbour(
                        i)[0] in subgraph._nodes_alias[i][0] 
                    except:
                        embed()


if __name__ == "__main__":
    TestGraph().test_setup_from_dataframe()
    # test_function()
