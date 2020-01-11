

# 1 Input test graph
#from gensim.models import Word2Vec
import n2v
from n2v import CSFGraph
import os

from n2v.word2vec import SkipGramWord2Vec

training_file = os.path.join(os.path.dirname(__file__), 'tests', 'data', 'karate.train')
training_file = '/home/robinp/PycharmProjects/IDG2KG-project-management/n2v/edgelist.txt'
#test_file= os.path.join(os.path.dirname(__file__), 'tests', 'data', 'karate.test')

print("Reading training file %s" % training_file)
training_graph = CSFGraph(training_file)
print(training_graph)
training_graph.print_edge_type_distribution()


p = 1
q = 1
gamma = 1
useGamma = False
hetgraph = n2v.hetnode2vec.N2vGraph(training_graph, p, q, gamma, useGamma)

walk_length = 80
num_walks = 25
walks = hetgraph.simulate_walks(num_walks, walk_length)
#walks = [map(str, walk) for walk in walks]
dimensions = 128
window_size = 10
workers = 8

worddictionary = training_graph.get_node_to_index_map()
reverse_worddictionary = training_graph.get_index_to_node_map()

numberwalks = []
for w in walks:
    nwalk = []
    for node in w:
        i = worddictionary[node]
        nwalk.append(i)
    numberwalks.append(nwalk)

model = SkipGramWord2Vec(numberwalks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_steps=100000)
model.train(display_step=1000)
output_filenname = 'disease.embedded'
model.write_embeddings(output_filenname)