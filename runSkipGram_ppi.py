import xn2v
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
import os


training_file = os.path.join(os.path.dirname(__file__),  'pos_train_edges')

g = CSFGraph(training_file)


p = 1
q = 1
gamma = 1
useGamma = False
graph = xn2v.hetnode2vec.N2vGraph(g, p, q, gamma, useGamma)

walk_length = 80
num_walks = 100
walks = graph.simulate_walks(num_walks, walk_length)
dimensions = 128
window_size = 10
workers = 8


worddictionary = g.get_node_to_index_map()
reverse_worddictionary = g.get_index_to_node_map()

walks_integer_nodes = []
for w in walks:
    nwalk = []
    for node in w:
        i = worddictionary[node]
        nwalk.append(i)
    walks_integer_nodes.append(nwalk)

model = SkipGramWord2Vec(walks_integer_nodes, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_steps=100)
model.train(display_step=2)











