from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v import LinkPrediction
import xn2v
import os
import logging

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "link_prediction.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
log.addHandler(handler)

pos_training_file = os.path.join(os.path.dirname(__file__),  'pos_train_edges')
pos_test_file = os.path.join(os.path.dirname(__file__),  'pos_test_edges')

neg_test_file = os.path.join(os.path.dirname(__file__),  'neg_test_edges')
neg_training_file = os.path.join(os.path.dirname(__file__),  'neg_train_edges')


pos_train_graph = CSFGraph(pos_training_file)
pos_test_graph = CSFGraph(pos_test_file)
neg_train_graph = CSFGraph(neg_training_file)
neg_test_graph = CSFGraph(neg_test_file)

p = 1
q = 1
gamma = 1
useGamma = False
hetgraph = xn2v.hetnode2vec.N2vGraph(pos_train_graph, p, q, gamma, useGamma)

walk_length = 80
num_walks = 100
dimensions = 128
window_size = 10
workers = 8
walks = hetgraph.simulate_walks(num_walks, walk_length)


worddictionary = pos_train_graph.get_node_to_index_map()
reverse_worddictionary = pos_train_graph.get_index_to_node_map()

walks_integer_nodes = []
for w in walks:
    nwalk = []
    for node in w:
        i = worddictionary[node]
        nwalk.append(i)
    walks_integer_nodes.append(nwalk)

model = SkipGramWord2Vec(walks_integer_nodes, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_steps=1000)
model.train(display_step=2)


output_filename = 'embedded_graph.embedded'
model.write_embeddings(output_filename)
path_to_embedded_graph = output_filename
edge_embedding_method = "hadamard"
lp = LinkPrediction(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph,
                            path_to_embedded_graph, edge_embedding_method)

#lp.output_diagnostics_to_logfile()
lp.predict_links()
lp.output_Logistic_Reg_results()
