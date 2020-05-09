import xn2v
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v.word2vec import ContinuousBagOfWordsWord2Vec
from xn2v.utils import write_embeddings
dir = '/home/peter/GIT/node2vec-eval'
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir,'tests/data/ppismall/pos_train_edges') # os.path.join(dir, 'pos_train_edges')

g = CSFGraph(training_file)


p = 1
q = 1
graph = xn2v.random_walk_generator.N2vGraph(g, p, q)
walk_length = 80
num_walks = 10
walks = graph.simulate_walks(num_walks, walk_length)

worddictionary = g.get_node_to_index_map()
reverse_worddictionary = g.get_index_to_node_map()

model = SkipGramWord2Vec(walks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_epochs=1)
model.train()
write_embeddings("pos_train_SG.embedded", model.embedding, reverse_worddictionary)


print("And now let's try CBOW")

model = ContinuousBagOfWordsWord2Vec(walks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary, num_epochs=1)
model.train()
write_embeddings("pos_train_CBOW.embedded", model.embedding, reverse_worddictionary)










