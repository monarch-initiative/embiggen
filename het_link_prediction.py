import argparse
import networkx as nx
import os
import logging
from xn2v import LinkPrediction




handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "link_prediction.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
log.addHandler(handler)


def parse_args():
    """
    parse args of link prediction
    """
    parser = argparse.ArgumentParser(description="Run link prediction.")

    parser.add_argument('--train', nargs='?',
                        help='Input graph path constructed from older files (training)')

    parser.add_argument('--test', nargs='?',
                        help='Input graph path containing the gene-disease edges')

    parser.add_argument('--embedded_train_graph', nargs='?',
                        help='The embedding  file of the old graph')

    parser.add_argument('--output_results', nargs='?',
                        help='Output file for results of link prediction')

    parser.add_argument('--output_edge_info_networks', nargs='?',
                        help='The information about all edges in all networks.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                            help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    """
    Reads the input networks (training graph and test graph) in networkx.
    """
    if args.weighted:
        g_train = nx.read_edgelist(args.train, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        g_test = nx.read_edgelist(args.test, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        # if the input graph was unweighted, assign 1 to all edges to avoid program errors
        g_train = nx.read_edgelist(args.train, nodetype=str, create_using=nx.DiGraph())
        g_test = nx.read_edgelist(args.test, nodetype=str, create_using=nx.DiGraph())
        for edge in g_train.edges():
            g_train[edge[0]][edge[1]]['weight'] = 1
        for edge in g_test.edges():
            g_test[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        # if the graph is not directd, then make the graph undirected
        g_train = g_train.to_undirected()
        g_test = g_test.to_undirected()

    return g_train, g_test




def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    log.debug("Reading training and test graphs")
    nx_g_train, nx_g_test = read_graph()
    path_to_embedded_graph = args.embedded_train_graph
    lp = LinkPrediction(nx_g_train, nx_g_test,path_to_embedded_graph)
    lp.output_diagnostics_to_logfile()
    lp.predict_links()





if __name__ == "__main__":
    log.debug("starting execution of het_link_prediction.py")
    args = parse_args()
    log.debug("Parameters: Input_training_graph: {}, test_graph: {}, "
              "embedded_training_graph: {}".format(args.train, args.test, args.embedded_train_graph))
    main(args)

