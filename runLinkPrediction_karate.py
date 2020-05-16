import embiggen
from embiggen import CSFGraph
from embiggen.word2vec import SkipGramWord2Vec
from embiggen.utils import write_embeddings
from embiggen import LinkPrediction
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
pos_train_file = os.path.join(current_dir, 'tests/data/karate/pos_train_edges')
pos_valid_file = os.path.join(current_dir, 'tests/data/karate/pos_validation_edges')
pos_test_file = os.path.join(current_dir, 'tests/data/karate/pos_test_edges')
neg_train_file = os.path.join(current_dir, 'tests/data/karate/neg_train_edges')
neg_valid_file = os.path.join(current_dir, 'tests/data/karate/neg_validation_edges')
neg_test_file = os.path.join(current_dir, 'tests/data/karate/neg_test_edges')

pos_train_graph = CSFGraph(pos_train_file)
pos_valid_graph = CSFGraph(pos_valid_file)
pos_test_graph = CSFGraph(pos_test_file)
neg_train_graph = CSFGraph(neg_train_file)
neg_valid_graph = CSFGraph(neg_valid_file)
neg_test_graph = CSFGraph(neg_test_file)

#parameters of the random walk
p = 1
q = 1
walk_length = 80
num_walks = 10

embed_graph = "pos_train_karate.embedded" #the embedded graph
edge_embed_method = "hadamard" #edge embedding method
classifier = "RF"# Random Forest:binary classifier in the link prediction
useValidation = True #Validation sets are
num_epochs = 2

#Graph (node) embeding using SkipGram as the word2vec model, with 2 epochs.
graph = embiggen.random_walk_generator.N2vGraph(pos_train_graph, p, q)
walks = graph.simulate_walks(num_walks, walk_length)
worddictionary = pos_train_graph.get_node_to_index_map()
reverse_worddictionary = pos_train_graph.get_index_to_node_map()
model = SkipGramWord2Vec(walks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_epochs=num_epochs)
model.train()
write_embeddings(embed_graph, model.embedding, reverse_worddictionary)

#Link prediction on the pos/neg train/valid/test sets using RF classifier
lp = LinkPrediction(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph,
                    embed_graph, edge_embed_method, classifier, useValidation)
lp.prepare_edge_and_node_labels()
lp.predict_links()
lp.output_classifier_results()









