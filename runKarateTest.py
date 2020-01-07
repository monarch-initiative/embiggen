## Demo


# 1 Input test graph
#from gensim.models import Word2Vec
import n2v
from n2v import LinkPrediction
from n2v import CSFGraph
import os

from n2v.word2vec import SkipGramWord2Vec

training_file = os.path.join(os.path.dirname(__file__), 'tests', 'data', 'karate.train')
test_file= os.path.join(os.path.dirname(__file__), 'tests', 'data', 'karate.test')

#g = nx.read_edgelist(training_file, nodetype=str, create_using=nx.DiGraph())

g = CSFGraph(training_file)

#for edge in g.edges():
#    g[edge[0]][edge[1]]['weight'] = 1

# Note -- weights of 1 added to karate files

#g_train = g.to_undirected()

p = 1
q = 1
gamma = 1
useGamma = False
hetgraph = n2v.hetnode2vec.Graph(g, p, q, gamma, useGamma)

walk_length = 80
num_walks = 100
walks = hetgraph.simulate_walks(num_walks, walk_length)
#walks = [map(str, walk) for walk in walks]
dimensions = 128
window_size = 10
workers = 8

print(walks)




if 34 in walks:
    print("BAD")
    exit(1)


worddictionary = g.get_node_to_index_map()
reverse_worddictionary = g.get_index_to_node_map()

numberwalks = []
for w in walks:
    nwalk = []
    for node in w:
        i = worddictionary[node]
        if i > 33:
            raise TypeError("BAD INDEX")
        nwalk.append(i)
    numberwalks.append(nwalk)


#model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iter)

"""
data,
                 worddictionary,
                 reverse_worddictionary,
                 learning_rate=0.1,
                 batch_size=128,
                 num_steps=3000000,
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=2,
                 skip_window=3,
                 num_skips=2,
                 num_sampled=64
"""


model = SkipGramWord2Vec(numberwalks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary)
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