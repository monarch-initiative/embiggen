###
#from Tiffany Callahan's github repository: https://github.com/callahantiff/owl-nets/blob/master/LinkPrediction.py
##Implementing 4 link prediction functions modfied according to CFSgraph class

###

import numpy as np  # type: ignore
import tensorflow as tf
def DegreeProduct(graph, node_1, node_2):
    ''' Function takes a CSF graph object and list of edges calculates the Degree Product or Preferential Attachment for these
    nodes given the structure of the graph.
    :param graph: CSFgraph  object
    :param node_1: one node of graph
    :param node_2: one node of a graph
    :return: Degree Product score of the nodes
    '''

    score = (graph.node_degree(node_1) * graph.node_degree(node_2))

    return score


def CommonNeighbors(graph, node_1, node_2):
    ''' Function takes a CSF graph object and list of edges calculates the Common Neighbors for these nodes given the
    structure of the graph.
    :param graph: CSFgraph object
    :param node_1: one node of graph
    :param node_2: one node of a graph
    :return: Common Neighbors score of the nodes
    '''

    node_1_neighbors = set(graph.neighbors(node_1))
    node_2_neighbors = set(graph.neighbors(node_2))

    if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
        score = 0.0

    elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
        score = 0.0

    else:
        n_intersection = node_1_neighbors.intersection(node_2_neighbors)
        score = float(len(n_intersection))

    return score

def Jaccard(graph, node_1, node_2):
    ''' Function takes a CFS graph object and list of edges calculates the Jaccard for these nodes given the
    structure of the graph.
    :param graph: CFS graph object
    :param node_1: one node of graph
    :param node_2: one node of a graph
    :return: The Jaccad score of two nodes
    '''

    node_1_neighbors = set(graph.neighbors(node_1))
    node_2_neighbors = set(graph.neighbors(node_2))

    if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
        score = 0.0

    elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
        score = 0.0

    else:
        n_intersection = set(node_1_neighbors.intersection(node_2_neighbors))
        n_union = set(node_1_neighbors.union(node_2_neighbors))
        score = float(len(n_intersection))/len(n_union)

    return score


def AdamicAdar(graph, node_1, node_2):
    ''' Function takes a CSF graph object and list of edges calculates the Adamic Adar for the nodes given the
    structure of the graph.
    :param graph: CSF graph object
    :param node_1: a node of the graph
    :param node_2: one node of a graph
    :return: AdamicAdar score of the nodes
    '''

    node_1_neighbors = set(graph.neighbors(node_1))
    node_2_neighbors = set(graph.neighbors(node_2))

    if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
        score = 0.0

    elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
        score = 0.0

    else:
        score = 0.0
        n_intersection = node_1_neighbors.intersection(node_2_neighbors)

        for c in n_intersection:
            score += 1/np.log(graph.node_degree(c))
    return score


#def get_cosine_sim(emb, valid_words, top_k):
   # norm = np.sqrt(np.sum(emb**2,axis=1,keepdims=True))
   # norm_emb = emb/norm
    #in_emb = norm_emb[valid_words,:]
   # similarity = np.dot(in_emb, np.transpose(norm_emb))
    #sorted_ind = np.argsort(-similarity, axis=1)[:,1:top_k+1]
    #return sorted_ind, valid_words



def cosine_similarity_tf(emb1, emb2):
    # cosin_similarity of two vectors X and Y with tensorflow:
    # cosine similarity = <X,Y> / ||X||||Y|| where <X,Y> is the dot product and ||X|| is sqrt(<X,X>)
    m = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
    m.update_state(emb1, emb2)
    return m.result().numpy()

