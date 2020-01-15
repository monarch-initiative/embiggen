import networkx as nx
import pandas as pd
import numpy as np

"""
For testing: generate a random graph with 100 nodes and 5000 edges between random nodes,
with random weights
"""

seed = 42   # for reproducibility
np.random.seed(seed)

num_nodes = 100
num_edges = 5000
graph = nx.complete_graph(num_nodes)

weights = np.random.rand(num_edges)
node1 = np.random.randint(1, num_nodes, num_edges)
node2 = np.random.randint(1, num_nodes, num_edges)
df = pd.DataFrame({'node1': node1, 'node2': node2, 'weight': weights})
df.to_csv("rand_" + str(num_nodes) + "nodes_" + str(num_edges) + "edges.graph",
          index=False, sep="\t", header=False)
