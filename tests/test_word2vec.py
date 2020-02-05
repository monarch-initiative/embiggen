from unittest import TestCase

from xn2v import CBOWBatcherListOfLists
from xn2v import CSFGraph
from xn2v import N2vGraph
from xn2v import ContinuousBagOfWordsWord2Vec



class TestTextEncoderSentences(TestCase):
    def setUp(self):
        list1 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        list2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        list3 = [112, 114, 116, 118, 1110, 1112, 1114, 1116, 1118, 1120]
        data = [list1, list2, list3]
        self.batcher = CBOWBatcherListOfLists(data, window_size=1)

    def test_ctor(self):
        self.assertIsNotNone(self.batcher)
        self.assertEqual(0, self.batcher.sentence_index)
        self.assertEqual(0, self.batcher.word_index)
        self.assertEqual(1, self.batcher.window_size)
        # span is 2*window_size +1
        self.assertEqual(3, self.batcher.span)
        self.assertEqual(3, self.batcher.sentence_count)
        self.assertEqual(10, self.batcher.sentence_len)
        # batch size is calculated as (sentence_len - span + 1)=10-3+1
        self.assertEqual(8, self.batcher.batch_size)

    def test_generate_batch(self):
        # The first batch is from the window [1, 3, 5] and [3, 5, 7]
        batch, labels = self.batcher.generate_batch()
        batch_shape = (8, 2)
        self.assertEqual(batch.shape, batch_shape)
        label_shape = (8, 1)
        self.assertEqual(labels.shape, label_shape)
        # batch represents the context words [[1,5],[3,7]]
        # label represents the center words [[3],[5]]
        self.assertEqual(1, batch[0][0])
        self.assertEqual(5, batch[0][1])
        self.assertEqual(3, labels[0])
        # now the second example
        self.assertEqual(3, batch[1][0])
        self.assertEqual(7, batch[1][1])
        self.assertEqual(5, labels[1])
        # get another batch. We expect the second list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(2, batch[0][0])
        self.assertEqual(6, batch[0][1])
        self.assertEqual(4, labels[0])
        # get another batch. We expect the third list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(112, batch[0][0])
        self.assertEqual(116, batch[0][1])
        self.assertEqual(114, labels[0])
        # get another batch. We expect to go back to the first list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(1, batch[0][0])
        self.assertEqual(5, batch[0][1])
        self.assertEqual(3, labels[0])


class TestCBOWconstruction(TestCase):
    """
    Use the test files in test/data/ppismall to construct a CBOW and test that we can get some random walks
    """
    def setUp(self):
        # neg_test = 'data/ppismall/neg_test_edges'
        # neg_train = 'data/ppismall/neg_train_edges'
        #pos_test = 'data/ppismall/pos_test_edges'
        pos_train = 'data/ppismall/pos_train_edges'
        training_graph = CSFGraph(pos_train)
        worddictionary = training_graph.get_node_to_index_map()
        reverse_worddictionary = training_graph.get_index_to_node_map()
        p = 1
        q = 1
        gamma = 1
        useGamma = False
        self.number_of_nodes_in_training = training_graph.node_count()
        self.n2v_graph = N2vGraph(training_graph, p, q, gamma, useGamma)
        self.walk_length = 10
        self.num_walks = 5
        walks = self.n2v_graph.simulate_walks(self.num_walks, self.walk_length)
        walks_integer_nodes = []
        for w in walks:
            nwalk = []
            for node in w:
                i = worddictionary[node]
                nwalk.append(i)
            walks_integer_nodes.append(nwalk)
        self.walks = walks_integer_nodes
        self.cbow = ContinuousBagOfWordsWord2Vec(self.walks, worddictionary=worddictionary,
                               reverse_worddictionary=reverse_worddictionary, num_steps=100)

    def test_number_of_walks_performed(self):

        # Expected number of walks is num_walks times the number of nodes in the training graph
        expected_n_walks = self.num_walks * self.number_of_nodes_in_training
        self.assertEqual(expected_n_walks, len(self.walks))

    def test_get_cbow_batch(self):
        self.assertIsNotNone(self.cbow)
        walk_count = 2
        num_skips = 1
        skip_window = 2
        batch = self.cbow.next_batch_from_list_of_lists(walk_count, num_skips, skip_window)
        self.assertIsNotNone(batch)