import xn2v
from xn2v import CSFGraph
import os
import unittest
from unittest import TestCase
from xn2v.word2vec import SkipGramWord2Vec


class TestSkipGramWord2Vec(TestCase):

    def test_embedding(self):
        training_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'karate.train'
        )
        output_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'disease.embedded'
        )
        training_graph = CSFGraph(training_file)
        training_graph.print_edge_type_distribution()

        p = 1
        q = 1
        gamma = 1
        useGamma = False
        hetgraph = xn2v.hetnode2vec.N2vGraph(
            training_graph, p, q, gamma, useGamma)

        walk_length = 80
        num_walks = 25
        walks = hetgraph.simulate_walks(num_walks, walk_length)

        worddictionary = training_graph.get_node_to_index_map()
        reverse_worddictionary = training_graph.get_index_to_node_map()

        numberwalks = []
        for w in walks:
            nwalk = []
            for node in w:
                i = worddictionary[node]
                nwalk.append(i)
            numberwalks.append(nwalk)

        model = SkipGramWord2Vec(numberwalks, worddictionary=worddictionary,
                                 reverse_worddictionary=reverse_worddictionary, num_steps=100)
        model.train(display_step=10)
        model.write_embeddings(output_file)
