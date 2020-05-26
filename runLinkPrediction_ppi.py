import argparse
from embiggen import CSFGraph, CooccurrenceEncoder
from embiggen import N2vGraph, GraphPartitionTransfomer
from embiggen.glove import GloVeModel
from embiggen.word2vec import SkipGramWord2Vec
from embiggen.word2vec import ContinuousBagOfWordsWord2Vec
from embiggen.utils import write_embeddings, serialize, deserialize
from embiggen.neural_networks import MLP, FFNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from cache_decorator import Cache
from sanitize_ml_labels import sanitize_ml_labels
from deflate_dict import deflate
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import time
from typing import Dict

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "link_prediction.log"))
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
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
                        default='output_results.csv',
                        help='path to the output file which contains results of link prediction')

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('edge_embed_method', nargs='?', default='hadamard',
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

    parser.add_argument('--num_epochs', default=1, type=int,
                        help='Number of training epochs')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='node2vec p hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='node2vec q hyperparameter. Default is 1.')

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


def get_embedding_model(
    walks: tf.RaggedTensor,
    pos_train: CSFGraph,
    w2v_model: str,
    embedding_size: int,
    context_window: int,
    num_epochs: int
):
    """Return selected embedding model.

    Parameters
    --------------------
    walks: tf.RaggedTensor,
    pos_train: CSFGraph,
    w2v_model: str,
    embedding_size: int,
    context_window: int,
    num_epochs: int,

    Returns
    ---------------------
    Return the selected embedding model.
    """

    worddictionary = pos_train.get_node_to_index_map()
    reverse_worddictionary = pos_train.get_index_to_node_map()

    w2v_model = w2v_model.lower()

    if w2v_model not in ("skipgram", "cbow", "glove"):
        raise ValueError('w2v_model must be "cbow", "skipgram" or "glove"')

    if w2v_model == "skipgram":
        logging.info("SkipGram analysis ")
        return SkipGramWord2Vec(walks,
                                worddictionary=worddictionary,
                                reverse_worddictionary=reverse_worddictionary, num_epochs=num_epochs)
    if w2v_model == "cbow":
        logging.info("CBOW analysis ")
        return ContinuousBagOfWordsWord2Vec(walks,
                                            worddictionary=worddictionary,
                                            reverse_worddictionary=reverse_worddictionary, num_epochs=num_epochs)
    logging.info("GloVe analysis ")
    n_nodes = pos_train.node_count()
    cencoder = CooccurrenceEncoder(walks, window_size=2, vocab_size=n_nodes)
    cooc_dict = cencoder.build_dataset()
    return GloVeModel(co_oc_dict=cooc_dict, vocab_size=n_nodes, embedding_size=embedding_size,
                      context_size=context_window, num_epochs=num_epochs)


def read_graphs(pos_train: str, pos_valid: str, pos_test: str, neg_train: str, neg_valid: str, neg_test: str):
    """
    Reads pos_train, pos_vslid, pos_test, neg_train train_valid and neg_test edges with CSFGraph
    :return: pos_train, pos_valid, pos_test, neg_train, neg_valid and neg_test graphs in CSFGraph format
    """
    start = time.time()

    pos_train_graph = CSFGraph(pos_train)
    pos_valid_graph = CSFGraph(pos_valid)
    pos_test_graph = CSFGraph(pos_test)
    neg_train_graph = CSFGraph(neg_train)
    neg_valid_graph = CSFGraph(neg_valid)
    neg_test_graph = CSFGraph(neg_test)
    end = time.time()
    logging.info(
        "reading input edge lists files: {} seconds".format(end-start))

    return pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph


def get_classifier_model(classifier: str, **kwargs: Dict):
    """Return choen classifier model.

    Parameters
    ------------------
    classifier:str,
        Chosen classifier model. Can either be:
            - LR for LogisticRegression
            - RF for RandomForestClassifier
            - MLP for Multi-Layer Perceptron
            - FFNN for Feed Forward Neural Network
    **kwargs:Dict,
        Keyword arguments to be passed to the constructor of the model.

    Raises
    ------------------
    ValueError,
        When given classifier model is not supported.

    Returns
    ------------------
    An instance of the selected model.
    """
    if classifier == "LR":
        return LogisticRegression(**kwargs)
    if classifier == "RF":
        return RandomForestClassifier(**kwargs)
    if classifier == "MLP":
        return MLP(**kwargs)
    if classifier == "FFNN":
        return FFNN(**kwargs)

    raise ValueError(
        "Given classifier model {} is not supported.".format(classifier)
    )


@Cache("embiggen_cache/{function_name}/{_hash}.pkl.gz")
def get_random_walks(graph: CSFGraph, p: float, q: float, num_walks: int, walk_length: int) -> tf.RaggedTensor:
    """Return a new N2vGraph trained on the provided graph.

    Parameters
    -------------------
    graph: CSFGraph,
    p: float,
    q: float,
    num_walks: int,
    walk_length: int

    Returns
    -------------------
    Return tf.RaggedTensor containing the random walks.
    """
    random_walker = N2vGraph(graph, p, q)
    return random_walker.simulate_walks(num_walks, walk_length)


def performance_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return performance report for given predictions and ground truths.

    Parameters
    ------------------------
    y_true: np.ndarray,
        The ground truth labels.
    y_pred: np.ndarray,
        The labels predicted by the classifier.

    Returns
    ------------------------
    Dictionary with the performance metrics, including AUROC, AUPRC, F1 Score,
    and accuracy.
    """
    metrics = roc_auc_score, average_precision_score, f1_score
    report = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in metrics
    }
    report[sanitize_ml_labels(accuracy_score.__name__)] = accuracy_score(
        y_true, np.round(y_pred).astype(int)
    )
    return report


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
        "context_window ={}, dimension={}, Validation={}".format(args.p, args.q, args.classifier,
                                                                 args.w2v_model, args.num_epochs,
                                                                 args.context_window, args.embedding_size,
                                                                 args.skipValidation))

    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = read_graphs(
        args.pos_train, args.pos_valid, args.pos_test, args.neg_train, args.neg_valid, args.neg_test
    )

    walks = get_random_walks(pos_train, args.p, args.q,
                             args.num_walks, args.walk_length)
    model = get_embedding_model(walks, pos_train, args.w2v_model,
                                args.embedding_size, args.context_window, args.num_epochs)
    start = time.time()
    model.train()
    end = time.time()
    logging.info(" learning: {} seconds ".format(end-start))

    transformer = GraphPartitionTransfomer(
        model.embedding, method=args.edge_embed_method)

    X_train, y_train = transformer.transform_edges(pos_train, neg_train)
    X_test, y_test = transformer.transform_edges(pos_test, neg_test)
    X_valid, y_valid = transformer.transform_edges(pos_valid, neg_valid)

    classifier_model = get_classifier_model(
        args.classifier,
        **(
            dict(input_shape=X_train.shape[1])
            if args.classifier in ("MLP", "FFNN")
            else {}
        )
    )

    classifier_model.fit(X_train, y_train)

    y_train_pred = classifier_model.predict(X_train)
    y_test_pred = classifier_model.predict(X_test)
    y_valid_pred = classifier_model.predict(X_valid)

    report = deflate(dict(
        train=performance_report(y_train, y_train_pred),
        test=performance_report(y_test, y_test_pred),
        valid=performance_report(y_valid, y_valid_pred),
    ))

    return pd.DataFrame({
        k: [v]
        for k, v in report.items()
    })


if __name__ == "__main__":
    args = parse_args()
    report = main(args)
    report.to_csv(args.output)
