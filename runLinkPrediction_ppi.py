import argparse

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



def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run link Prediction.")

    parser.add_argument('--pos_train', nargs='?', default='pos_train_edges',
                        help='Input positive training edges path')

    parser.add_argument('--pos_test', nargs='?', default='pos_test_edges',
                        help='Input positive test edges path')

    parser.add_argument('--neg_train', nargs='?', default='neg_train_edges',
                        help='Input negative training edges path')

    parser.add_argument('--neg_test', nargs='?', default='neg_test_edges',
                        help='Input negative test edges path')

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('--edge_embed_method', nargs='?', default='hadamard',
                        help='Embeddings embedding method of the positive training graph. '
                             'It can be hadamard, weightedL1, weightedL2 or average')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--gamma', type=float, default=1,
                        help='hyperparameter for jumping from one network to another network '
                             'in heterogeneous graphs. Default is 1.')

    parser.add_argument('--useGamma', dest='useGamma', action='store_false', help="True if the graph is heterogeneous,"
                                                                                  "False if the graph is homogeneous.")

    parser.add_argument('--classifier', nargs='?', default='LR',
                        help="Binary classifier for link prediction, it should be either LR, RF or SVM")

    return parser.parse_args()


def learn_embeddings(walks, pos_train_graph):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    worddictionary = pos_train_graph.get_node_to_index_map()
    reverse_worddictionary = pos_train_graph.get_index_to_node_map()

    walks_integer_nodes = []
    for w in walks:
        nwalk = []
        for node in w:
            i = worddictionary[node]
            nwalk.append(i)
        walks_integer_nodes.append(nwalk)

    model = SkipGramWord2Vec(walks_integer_nodes, worddictionary=worddictionary,
                             reverse_worddictionary=reverse_worddictionary, num_steps=100)
    model.train(display_step=2)

    model.write_embeddings(args.embed_graph)

def linkpred(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph):
    """
    :param pos_train_graph: positive training graph
    :param pos_test_graph: positive test graph
    :param neg_train_graph: negative training graph
    :param neg_test_graph: negative test graph
    :return: Metrics of logistic regression as the results of link prediction
    """
    lp = LinkPrediction(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph,
                        args.embed_graph, args.edge_embed_method)

    lp.prepare_lables_test_training()
    lp.predict_links(args.classifier)
    lp.output_classifier_results()

def read_graphs():
    """
    Reads pos_train, pos_test, neg_train and neg_test edges with CSFGraph
    :return: pos_train, pos_test, neg_train and neg_test graphs in CSFGraph format
    """

    pos_train_graph = CSFGraph(args.pos_train)
    pos_test_graph = CSFGraph(args.pos_test)
    neg_train_graph = CSFGraph(args.neg_train)
    neg_test_graph = CSFGraph(args. neg_test)
    return pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph


def main(args):
    """
    The input files are positive training, positive test, negative training and negative test edges. The code
    reads the files and create graphs in CSFGraph format. Then, the positive training graph is embedded.
    Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Result of link prediction using logistic regression. TODO: implement RF and SVM and add the option of CBOW for word2vec
    """
    pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph = read_graphs()
    pos_train_g = xn2v.hetnode2vec.N2vGraph(pos_train_graph,  args.p, args.q, args.gamma, args.useGamma)
    walks = pos_train_g.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, pos_train_graph)
    linkpred(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph)


if __name__ == "__main__":
    args = parse_args()
    main(args)
