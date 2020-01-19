import unittest
from unittest import TestCase
import n2v
from n2v import LinkPrediction
from n2v import CSFGraph
import os

from n2v.word2vec import SkipGramWord2Vec

training_file = os.path.join(os.path.dirname(
    __file__), 'data', 'karate.train')
test_file = os.path.join(os.path.dirname(
    __file__), 'data', 'karate.test')


class TestKarate(TestCase):

    def test_karate(self):
        training_graph = CSFGraph(training_file)
        p = 1
        q = 1
        gamma = 1
        useGamma = False
        hetgraph = n2v.hetnode2vec.N2vGraph(
            training_graph, p, q, gamma, useGamma)

        walk_length = 80
        num_walks = 20
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
                                 reverse_worddictionary=reverse_worddictionary, num_steps=1000)
        model.train(display_step=100)
        output_filenname = 'karate.embedded'
        model.write_embeddings(output_filenname)

        test_graph = CSFGraph(test_file)
        path_to_embedded_graph = output_filenname
        parameters = {'edge_embedding_method': "hadamard",
                      'portion_false_edges': 1}

        lp = LinkPrediction(training_graph, test_graph,
                            path_to_embedded_graph, params=parameters)

        # lp.output_diagnostics_to_logfile()
        lp.predict_links()
        lp.output_Logistic_Reg_results()
