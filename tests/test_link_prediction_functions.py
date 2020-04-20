from unittest import TestCase
import os.path
from xn2v import CSFGraph
from xn2v import link_prediction_functions
import numpy as np


class TestCSFGraph(TestCase):
    def setUp(self):
        inputfile = os.path.join(os.path.dirname(
            __file__), 'data', 'small_graph.txt')
        g = CSFGraph(inputfile)
        self.g = g
        str(g)

    def test_notnull(self):
        self.assertIsNotNone(self.g)

    def test_DegreeProduct(self):
        ""
        #Test the Degree product score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        #neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. Thus the score = 5 * 4 = 20
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(20, score)

    def test_CommonNeighbors(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        #neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection is {g3, p1}. So, score is 2.
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(2, score)

    def test_Jaccard(self):
        ""
        # Test the Jaccard score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        # neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection = {g3, p1}.
        # The union = {g1,g2,g3,g4,d3,p1,p3}. So, score is 2/7.
        score = link_prediction_functions.Jaccard(self.g, node_1, node_2)
        self.assertEqual(2.0/7.0, score)

    def test_AdamicAdar(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        # neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection = {g3, p1}.
        #Neighbors of g3 = {g1,g2,g4,p3,d1}, neighbors of p1 = {g1,g2,p2,p3}
        # The score = (1/log 5) + (1/ log 4) = (1/1.609) + (1/ 1.386) = 0.621 + 0.721 = 1.342
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual((1.0/np.log(5)) + (1.0/np.log(4)), score)


