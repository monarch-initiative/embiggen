import argparse
from embiggen import CSFGraph, CooccurrenceEncoder
from embiggen import N2vGraph
from embiggen.glove import GloVeModel
from embiggen.word2vec import SkipGramWord2Vec
from embiggen.word2vec import ContinuousBagOfWordsWord2Vec
from embiggen import LinkPrediction
from embiggen.utils import write_embeddings, serialize, deserialize
import os
import logging
import time

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE","link_prediction.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
log.addHandler(handler)


def parse_args():
    """Parses arguments.

    """
    parser = argparse.ArgumentParser(description="Run link Prediction.")

    parser.add_argument('--pos_train', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_train_edges_max_comp_graph',
                        help='Input positive training edges path')

    parser.add_argument('--pos_valid', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_validation_edges_max_comp_graph',
                       help='Input positive validation edges path')

    parser.add_argument('--pos_test', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_test_edges_max_comp_graph',
                        help='Input positive test edges path')

    parser.add_argument('--neg_train', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_train_edges_max_comp_graph',
                        help='Input negative training edges path')

    parser.add_argument('--neg_valid', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_validation_edges_max_comp_graph',
                        help='Input negative validation edges path')

    parser.add_argument('--neg_test', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_test_edges_max_comp_graph',
                        help='Input negative test edges path')

    parser.add_argument('--output', nargs='?',
                        default='output_results',
                        help='path to the output file which contains results of link prediction')

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('--edge_embed_method', nargs='?', default='hadamard',
                        help='Embeddings embedding method of the positive training graph. '
                             'It can be hadamard, weightedL1, weightedL2 or average')

    parser.add_argument('--embedding_size', type=int, default=200,
                        help='Number of dimensions which is size of the embedded vectors. Default is 200.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--context_window', type=int, default=3,
                        help='Context size for optimization. Default is 3.')

    parser.add_argument('--num_skips', type=int, default=2,
                        help='Default is 2.')

    parser.add_argument('--num_epochs', default=1, type=int,
                        help='Number of training epochs')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='node2vec p hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='node2vec q hyperparameter. Default is 1.')

    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate for GD. Default is 0.1.')

    parser.add_argument('--num_nearest_neighbors', type=int, default=16,
                        help='Number of nearest neighbors to each word. Default is 16. A valid number is '
                             'less than or equal to 16.')

    parser.add_argument('--classifier', nargs='?', default='LR',
                        help="Binary classifier for link prediction, it should be either LR, RF, SVM, MLP, FFNN")

    parser.add_argument('--w2v_model', nargs='?', default='Skipgram',
                        help="word2vec model (Skipgram, CBOW, GloVe)")

    parser.add_argument('--skipValidation', dest='skipValidation', action='store_true',
                        help="Boolean specifying presence of validation sets or not. Default is validation sets are provided.")
    parser.set_defaults(skipValidation=False)

    parser.add_argument('--random_walks', type=str,
                        help='Use a cached version of random walks. \
                        (Note: This assumes that --pos_train is the same as the one used to build the cached version)')

    parser.add_argument('--cache_random_walks', action='store_true',
                        help='Cache the random walks generated from pos_train CsfGraph. \
                        (--random_walks argument must be defined)')

    parser.add_argument('--use_cached_random_walks', action='store_true',
                        help='Use the cached version of random walks. \
                        (--random_walks argument must be defined)\
                        (Note: This assumes that --pos_train is the same as the one used to build the cached version)')

    return parser.parse_args()


def learn_embeddings(walks, pos_train_graph, w2v_model):
    """
    Learn embeddings by optimizing the Glove, Skipgram or CBOW objective using SGD.
    """

    worddictionary = pos_train_graph.get_node_to_index_map()
    reverse_worddictionary = pos_train_graph.get_index_to_node_map()

    if w2v_model.lower() == "skipgram":
        logging.info("SkipGram analysis ")
        model = SkipGramWord2Vec(walks,
                                 worddictionary=worddictionary,
                                 reverse_worddictionary=reverse_worddictionary, num_epochs=args.num_epochs,
                                 learning_rate= args.learning_rate,
                                 embedding_size=args.embedding_size, context_window=args.context_window,
                                 num_skips=args.num_skips)
    elif w2v_model.lower() == "cbow":
        logging.info("CBOW analysis ")
        model = ContinuousBagOfWordsWord2Vec(walks,
                                             worddictionary=worddictionary,
                                             reverse_worddictionary=reverse_worddictionary, num_epochs=args.num_epochs,
                                             learning_rate= args.learning_rate,
                                             embedding_size=args.embedding_size, context_window=args.context_window,
                                             num_skips=args.num_skips)
    elif w2v_model.lower() == "glove":
        logging.info("GloVe analysis ")
        n_nodes = pos_train_graph.node_count()
        cencoder = CooccurrenceEncoder(walks, window_size=2, vocab_size=n_nodes)
        cooc_dict = cencoder.build_dataset()
        model = GloVeModel(co_oc_dict=cooc_dict, vocab_size=n_nodes, embedding_size=args.embedding_size,
                           context_size=args.context_window, num_epochs=args.num_epochs,
                           learning_rate=args.learning_rate)
    else:
        raise ValueError('w2v_model must be "cbow", "skipgram" or "glove"')

    model.add_display_words(args.num_nearest_neighbors)

    model.train()

    write_embeddings(args.embed_graph, model.embedding, reverse_worddictionary)


def linkpred(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph):
    """
    :param pos_train_graph: positive training graph
    :param pos_valid_graph: positive validation graph
    :param pos_test_graph: positive test graph
    :param neg_train_graph: negative training graph
    :param neg_valid_graph: negative validation graph
    :param neg_test_graph: negative test graph
    :return: Metrics of logistic regression as the results of link prediction
    """
    lp = LinkPrediction(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph,
                                      neg_valid_graph, neg_test_graph, args.embed_graph, args.edge_embed_method, args.classifier,
                                      args.skipValidation, args.output)

    lp.prepare_edge_and_node_labels()
    lp.predict_links()
    lp.output_classifier_results()
    #lp.output_edge_node_information()
    #lp.predicted_ppi_links()
    #lp.predicted_ppi_non_links()

def read_graphs():
    """
    Reads pos_train, pos_vslid, pos_test, neg_train train_valid and neg_test edges with CSFGraph
    :return: pos_train, pos_valid, pos_test, neg_train, neg_valid and neg_test graphs in CSFGraph format
    """
    start = time.time()

    pos_train_graph = CSFGraph(args.pos_train)
    pos_valid_graph = CSFGraph(args.pos_valid)
    pos_test_graph = CSFGraph(args.pos_test)
    neg_train_graph = CSFGraph(args.neg_train)
    neg_valid_graph = CSFGraph(args.neg_valid)
    neg_test_graph = CSFGraph(args.neg_test)
    end = time.time()
    logging.info("reading input edge lists files: {} seconds".format(end-start))

    return pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph


def main(args):
    """
    The input files are positive training, positive test, negative training and negative test edges. The code
    reads the files and create graphs in CSFGraph format. Then, the positive training graph is embedded.
    Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Result of link prediction
    """
    logging.info(
        " p={}, q={}, classifier= {}, word2vec_model={}, num_epochs={}, "
        "context_window ={}, dimension={}, skipValidation={}, num_skips={}, learning_rate={}".
                                                                            format(args.p, args.q, args.classifier,
                                                                            args.w2v_model, args.num_epochs,
                                                                            args.context_window, args.embedding_size,
                                                                            args.skipValidation, args.num_skips,
                                                                            args.learning_rate))

    pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph = read_graphs()
    if args.use_cached_random_walks and args.random_walks:
        # restore post_train_g from cache
        logging.info(f"Restore random walks from {args.random_walks}")
        start = time.time()
        pos_train_g = deserialize(args.random_walks)
        end = time.time()
        logging.info(" de-serializing: {} seconds ".format(end - start))

    else:
        # generate pos_train_g and simulate walks
        pos_train_g = N2vGraph(pos_train_graph, args.p, args.q)
        start = time.time()
        pos_train_g.simulate_walks(args.num_walks, args.walk_length, args.use_cached_random_walks)
        end = time.time()
        logging.info("simulating walks: {} seconds".format(end - start))

    if args.cache_random_walks and args.random_walks:
        logging.info(f"Caching random walks to {args.random_walks}")
        start = time.time()
        serialize(pos_train_g, args.random_walks)
        end = time.time()
        logging.info(" serializing: {} seconds ".format(end - start))

    walks = pos_train_g.random_walks_map[(args.num_walks, args.walk_length)]

    start = time.time()
    learn_embeddings(walks, pos_train_graph, args.w2v_model)
    end = time.time()
    logging.info(" learning: {} seconds ".format(end-start))

    start = time.time()
    linkpred(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph)
    end = time.time()
    logging.info("link prediction: {} seconds".format(end-start))

if __name__ == "__main__":
    args = parse_args()
    main(args)
