from unittest import TestCase
import os.path
from xn2v import CSFGraph
from xn2v import link_prediction_functions
import numpy as np
import math



class TestlinkPredictionScores_1(TestCase):
    def setUp(self):
        inputfile = os.path.join(os.path.dirname(
            __file__), 'data', 'small_graph.txt')
        g = CSFGraph(inputfile)
        self.g = g
        str(g)

    def test_notnull(self):
        self.assertIsNotNone(self.g)

    def test_DegreeProduct_1(self):
        ""
        #Test the Degree product score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        #neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. Thus the score = 5 * 4 = 20
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(20, score)

    def test_DegreeProduct_2(self):
        ""
        # Test the Degree product score of two nodes
        node_1 = "g4"
        node_2 = "d1"
        # neighbors of "g4" = {g1,g3,p4,d2}, neighbors of "d1" = {d1,d2, g3}. Thus the score = 4 * 3 = 12
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(12, score)

    def test_CommonNeighbors_1(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        #neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection is {g3, p1}. So, score is 2.
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(2, score)

    def test_CommonNeighbors_2(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g4"
        node_2 = "d1"
        # neighbors of "g4" = {g1,g3,p4,d2}, neighbors of "d1" = {d1,d2, g3}. The intersection is {d2,g3}
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(2, score)

    def test_Jaccard_1(self):
        ""
        # Test the Jaccard score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        # neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection = {g3, p1}.
        # The union = {g1,g2,g3,g4,d3,p1,p3}. So, score is 2/7.
        score = link_prediction_functions.Jaccard(self.g, node_1, node_2)
        self.assertEqual(2.0/7.0, score)

    def test_Jaccard_2(self):
        ""
        # Test the Jaccard score of two nodes
        node_1 = "g4"
        node_2 = "d1"
        # neighbors of "g4" = {g1,g3,p4,d2}, neighbors of "d1" = {d1,d2, g3}. The intersection is {d2,g3}.
        # The union = {g1,g3,p4,d1,d2}. So, score is 2/5.
        score = link_prediction_functions.Jaccard(self.g, node_1, node_2)
        self.assertEqual(2.0 / 5.0, score)

    def test_AdamicAdar_1(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g1"
        node_2 = "g2"
        # neighbors of "g1" = {g2,g3,g4,p1,d3}, neighbors of "g2" = {g1,g3,p1,p3}. The intersection = {g3, p1}.
        #Neighbors of g3 = {g1,g2,g4,p3,d1}, neighbors of p1 = {g1,g2,p2,p3}
        # The score = (1/log 5) + (1/ log 4) = (1/1.609) + (1/ 1.386) = 0.621 + 0.721 = 1.342
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual((1.0/np.log(5)) + (1.0/np.log(4)), score)

    def test_AdamicAdar_2(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g4"
        node_2 = "d1"
        # neighbors of "g4" = {g1,g3,p4,d2}, neighbors of "d1" = {d1,d2, g3}. The intersection is {d2,g3}.
        # #Neighbors of d2 = {g4,d3,d1}, neighbors of g3 = {g1,g2,g4,d1,p3}
        # The score = (1/log 3) + (1/ log 5) = (1/1.098) + (1/ 1.609) =  0.910+ 0.621 = 1.531
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual((1.0/np.log(3)) + (1.0/np.log(5)), score)

class TestlinkPredictionScores_2(TestCase):
    def setUp(self):
        inputfile = os.path.join(os.path.dirname(
            __file__), 'data', 'small_het_graph_edges.tsv')
        g = CSFGraph(inputfile)
        self.g = g
        str(g)

    def test_notnull(self):
        self.assertIsNotNone(self.g)

    def test_DegreeProduct_1(self):
        ""
        # Test the Degree product score of two nodes
        node_1 = "g1"
        node_2 = "d2"
        # neighbors of "g1" = {g0,g2,g3,...,g100,p1,d1}, neighbors of "d2" = {d1}. Thus the score = 102 * 1 = 102
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(102, score)

    def test_DegreeProduct_2(self):
        ""
        # Test the Degree product score of two nodes
        node_1 = "g3"
        node_2 = "p3"
        # neighbors of "g3" = {g1}, neighbors of "p3" = {p1}. Thus the score = 1 * 1 = 1
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(1, score)

    def test_DegreeProduct_3(self):
        ""
        # Test the Degree product score of two nodes
        node_1 = "g3"
        node_2 = "g4"
        # neighbors of "g3" = {g1}, neighbors of "g4" = {g1}. Thus the score = 1 * 1 = 1
        score = link_prediction_functions.DegreeProduct(self.g, node_1, node_2)
        self.assertEqual(1, score)

    def test_CommonNeighbors_1(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g1"
        node_2 = "d2"
        # neighbors of "g1" = {g0,g2,g3,...,g100,p1,d1}, neighbors of "d2" = {d1}. The intersection is d1. So, the score is 1.
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(1, score)

    def test_CommonNeighbors_2(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g3"
        node_2 = "p3"
        # neighbors of "g3" = {g1}, neighbors of "p3" = {p1}. Thus the score = 0
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(0, score)

    def test_CommonNeighbors_3(self):
        ""
        # Test the Common Neighbor score of two nodes
        node_1 = "g3"
        node_2 = "g4"
        # neighbors of "g3" = {g1}, neighbors of "g4" = {g1}. Thus the score = 1
        score = link_prediction_functions.CommonNeighbors(self.g, node_1, node_2)
        self.assertEqual(1, score)

    def test_Jaccard_1(self):
        ""
        # Test the Jaccard score of two nodes
        node_1 = "g1"
        node_2 = "d2"
        # neighbors of "g1" = {g0,g2,g3,...,g100,p1,d1}, neighbors of "d2" = {d1}.
        # The intersection is d1. .
        # The union = {g0,g2,g3,..., g100,p1,d1}. So, score is 1/102.
        score = link_prediction_functions.Jaccard(self.g, node_1, node_2)
        self.assertEqual(1.0/102.0, score)


    def test_Jaccard_3(self):
        ""
        # Test the Jaccard score of two nodes
        node_1 = "g3"
        node_2 = "g4"
        # neighbors of "g3" = {g1}, neighbors of "g4" = {g1}. The intersection is {g1}.
        # The union is {g1}. So, the score is 1
        score = link_prediction_functions.Jaccard(self.g, node_1, node_2)
        self.assertEqual(1, score)

    def test_AdamicAdar_1(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g1"
        node_2 = "d2"
        # neighbors of "g1" = {g0,g2,g3,...,g100,p1,d1}, neighbors of "d2" = {d1}.
        # The intersection is d1.
        # #Neighbors of d1 = {g1,d2,..,d20},
        # The score = (1/log 20) = 0.333
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual(1.0/np.log(20), score)


    def test_AdamicAdar_2(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g3"
        node_2 = "p3"
        # neighbors of "g3" = {g1}, neighbors of "p3" = {p1}. The intersection is {}.
        # The score =0
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual(0, score)

    def test_AdamicAdar_3(self):
        ""
        # Test the AdamicAdar score of two nodes
        node_1 = "g3"
        node_2 = "g4"
        # neighbors of "g3" = {g1}, neighbors of "g4" = {g1}. The intersection is {g1}.
        #neighbours of g1 = {g0, g2, ..., g100, p1,d1}. Score = 1/ log(102)
        score = link_prediction_functions.AdamicAdar(self.g, node_1, node_2)
        self.assertAlmostEqual(1.0/np.log(102.0), score)

class TestCosieSimilarity(TestCase):
    #
    #Test the cosine similarity of two vectors using tensorflow.
    #
    def test_cos_sim_tf_1(self):
        X = np.array([1,2])
        Y = np.array([3,4])
        cos_sim = link_prediction_functions.cosine_similarity_tf(X, Y)
        #<X,Y> = 1*3 + 2*4 = 11
        #sqrt(<X,X>)= sqrt(1+4), sqrt(<Y,Y>)= sqrt(9+16)=5. Thus cos_sim= 11/ 5*sqrt(5)
        estimate = round(11/(5*math.sqrt(5)), 7)#round the number to 7 decimals
        self.assertAlmostEqual(estimate, cos_sim)

    def test_cos_sim_tf_2(self):
        X = np.array([1,2,0])
        Y = np.array([3,4,1])
        cos_sim = link_prediction_functions.cosine_similarity_tf(X, Y)
        # <X,Y> = 1*3 + 2*4 + 0*1 = 11
        # sqrt(<X,X>)= sqrt(1+4), sqrt(<Y,Y>)= sqrt(9+16+1)=sqrt(26). Thus cos_sim= 11/ sqrt(26)*sqrt(5)
        estimate = round(11.0/(math.sqrt(26)*math.sqrt(5)), 7)#round the number to 7 decimals
        self.assertAlmostEqual(estimate, cos_sim)


    def test_cos_sim_tf_3(self):
        X = np.array([1, 0, 0])
        Y = np.array([3, 4, 0])
        cos_sim = link_prediction_functions.cosine_similarity_tf(X,Y)
        # <X,Y> = 1*3 =3
        # sqrt(<X,X>)= 1, sqrt(<Y,Y>)= sqrt(9+16)=5. Thus cos_sim= 3/5=0.6
        self.assertAlmostEqual(0.6, cos_sim)
