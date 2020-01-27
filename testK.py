import xn2v
from xn2v import CSFGraph
from xn2v.kW2V import kWord2Vec

training_file = 'tests/data/rand_100nodes_5000edges.graph'
num_walks = 20
walk_length = 80
p = 1
q = 1
gamma = 1
num_steps = 1
use_gamma = False

training_graph = CSFGraph(training_file)
print(training_graph)
training_graph.print_edge_type_distribution()

hetgraph = xn2v.hetnode2vec.N2vGraph(training_graph, p, q, gamma, use_gamma)
walks = hetgraph.simulate_walks(num_walks, walk_length)
worddictionary = training_graph.get_node_to_index_map()
reverse_worddictionary = training_graph.get_index_to_node_map()

numberwalks = []
for w in walks:
    nwalk = []
    for node in w:
        i = worddictionary[node]
        nwalk.append(i)
    numberwalks.append(nwalk)

model = kWord2Vec(numberwalks)
model.train()