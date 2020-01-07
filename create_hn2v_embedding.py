"""
Implementation of heterogeneous node2vec algorithm.
Based on the reference implementation of node2vec by Aditya Grover
and adapted from code at https://github.com/aditya-grover/node2vec
"""

import argparse
import networkx as nx
from gensim.models import Word2Vec
import hn2v
import os
import logging
log = logging.getLogger("hn2v.log")

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "hn2v.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
log.addHandler(logging.StreamHandler())


def parse_args():
	"""
	parse args of node2vec
	"""
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.train', help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb', help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128, help = 'Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--gamma', type=float, default=1, help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def output_args_to_logger(args):
	log.info("Input: {}".format(args.input))
	log.info("output: {}".format(args.output))
	log.info("weighted: {}".format(args.weighted))


def read_graph():
	"""
	Reads the input network in networkx.
	"""
	if args.weighted:
		g = nx.read_edgelist(args.input, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
	else:
		g = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
		for edge in g.edges():
			g[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		g = g.to_undirected()

	return g


def learn_embeddings(walks):
	"""
	Learn embeddings by optimizing the Skipgram objective using SGD.
	"""
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
					iter=args.iter)

	#model.wv.save_word2vec_format(args.output)# TODO:python 3 and more?
	model.save_word2vec_format(args.output) #python 2.7
	return


def main(args):
	"""
	Pipeline for representational learning for all nodes in a graph.
	"""
	nx_g = read_graph()
	log.debug("Extracted graph with {}".format(nx_g))
	g = hn2v.hetnode2vec.Graph(nx_g, args.directed, args.p, args.q, args.gamma, True)  # gamma is a parameter when we traverse
	# from one nodetype to another nodetype. Change True to False if you don't want to use the modified get-alias_adgeHN2V
	log.info("Done: preprocess transition probabilities ")
	walks = g.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)


if __name__ == "__main__":
	log.debug("starting execution of run_hn2v.py")
	args = parse_args()
	output_args_to_logger(args)
	main(args)
