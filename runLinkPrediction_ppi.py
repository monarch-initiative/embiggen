import logging
import sys
import tempfile
from urllib.request import urlopen



import xn2v
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v import LinkPrediction



import os


training_file = os.path.join(os.path.dirname(__file__),  'pos_train_edges')
#training_file = os.path.join(os.path.dirname(__file__),  'karate.train')

#test_file= os.path.join(os.path.dirname(__file__), 'data', 'karate.test')


g = CSFGraph(training_file)

#for edge in g.edges():
#    g[edge[0]][edge[1]]['weight'] = 1

# Note -- weights of 1 added to karate files

#g_train = g.to_undirected()

p = 1
q = 1
gamma = 1
useGamma = False
hetgraph = xn2v.hetnode2vec.N2vGraph(g, p, q, gamma, useGamma)

walk_length = 80
num_walks = 100
walks = hetgraph.simulate_walks(num_walks, walk_length)
#walks = [map(str, walk) for walk in walks]
dimensions = 128
window_size = 10
workers = 8


worddictionary = g.get_node_to_index_map()
reverse_worddictionary = g.get_index_to_node_map()



model = SkipGramWord2Vec(walks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary)
model.train(display_step=2)

"""
NEEDS TO BE REFACTORED
output = 'karate.embedded'
#model.save_word2vec_format(output)  # python 2.7
model.wv.save_word2vec_format(output)# TODO:python 3 and more?

g_train = nx.read_edgelist(training_file, nodetype=str, create_using=nx.DiGraph())
g_test = nx.read_edgelist(test_file, nodetype=str, create_using=nx.DiGraph())
for edge in g_train.edges():
    g_train[edge[0]][edge[1]]['weight'] = 1
for edge in g_test.edges():
    g_test[edge[0]][edge[1]]['weight'] = 1

#g_train = g_train.to_undirected()
#g_test = g_test.to_undirected()
path_to_embedded_graph = output
parameters = {'edge_embedding_method': "hadamard",
              'portion_false_edges': 2}

lp = LinkPrediction(g_train, g_test, path_to_embedded_graph, params=parameters)
lp.output_diagnostics_to_logfile()
lp.predict_links()
lp.output_Logistic_Reg_results()
"""









