import logging
import os
import sys
import tempfile
from urllib.request import urlopen

import click as click
from click import get_os_args

import xn2v
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v import LinkPrediction

@click.group()
def cli():
    pass

@cli.command()
@click.option("training_file", "-t", type=click.Path(exists=True), required=True)
@click.option("output_file", "-o", default='disease.embedded')
@click.option("p", "-p", type=int, default=1)
@click.option("q", "-q", type=int, default=1)
@click.option("gamma", "-g", type=int, default=1)
@click.option("use_gamma", "-u", is_flag=True, default=False)
@click.option("walk_length", "-w", type=int, default=80)
@click.option("num_walks", "-n", type=int, default=25)
@click.option("dimensions", "-d", type=int, default=128)
@click.option("window_size", "-w", type=int, default=10)
@click.option("workers", "-r", type=int, default=8)
@click.option("num_steps", "-s", type=int, default=100000)
@click.option("display_step", "-d", type=int, default=1000)
def disease_gene_embeddings(training_file, output_file, p, q, gamma, use_gamma,
                            walk_length, num_walks, dimensions, window_size, workers,
                            num_steps, display_step):
    """
    Generate disease gene embeddings
    """
    logging.basicConfig(level=logging.INFO)
    print("Reading training file %s" % training_file)
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

    model = SkipGramWord2Vec(numberwalks, worddictionary=worddictionary,
                             reverse_worddictionary=reverse_worddictionary,
                             num_steps=num_steps)
    model.train(display_step=display_step)
    model.write_embeddings(output_file)

@cli.command()
@click.option("test_file", "-t", type=click.Path(exists=True), required=True)
@click.option("training_file", "-r", type=click.Path(exists=True), required=True)
@click.option("embedded_graph", "-e", type=click.Path(exists=True), required=True)
@click.option("edge_embedding_method", "-m", default="hadamard")
@click.option("portion_false_edges", "-p", default=1)
def disease_link_prediction(test_file, training_file, embedded_graph,
                            edge_embedding_method, portion_false_edges):
    """
    Predict disease links
    """
    test_graph = CSFGraph(test_file)
    training_graph = CSFGraph(training_file)

    parameters = {'edge_embedding_method': edge_embedding_method,
                  'portion_false_edges': portion_false_edges}

    lp = LinkPrediction(training_graph, test_graph, embedded_graph, params=parameters)
    lp.predict_links()
    lp.output_Logistic_Reg_results()
    training_graph = CSFGraph(training_file)


@cli.command()
@click.option("training_file", "-t", type=click.Path(exists=True), required=True,
              default="tests/data/karate.train")
@click.option("test_file", "-t", type=click.Path(exists=True), required=True,
              default="tests/data/karate.test")
@click.option("output_file", "-o", default='karate.output')
@click.option("p", "-p", type=int, default=1)
@click.option("q", "-q", type=int, default=1)
@click.option("gamma", "-g", type=int, default=1)
@click.option("use_gamma", "-u", is_flag=True, default=False)
@click.option("walk_length", "-w", type=int, default=80)
@click.option("num_walks", "-n", type=int, default=25)
def karate_test(training_file, test_file, output_file, p, q, gamma, use_gamma,
                    walk_length, num_walks):
    training_graph = CSFGraph(training_file)
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

    model = SkipGramWord2Vec(numberwalks, worddictionary=worddictionary,
                             reverse_worddictionary=reverse_worddictionary,
                             num_steps=1000)
    model.train(display_step=100)
    output_filenname = 'karate.embedded'
    model.write_embeddings(output_filenname)

    test_graph = CSFGraph(test_file)
    path_to_embedded_graph = output_filenname
    parameters = {'edge_embedding_method': "hadamard",
                  'portion_false_edges': 1}

    lp = LinkPrediction(training_graph, test_graph, path_to_embedded_graph,
                        params=parameters)

    lp.predict_links()
    lp.output_Logistic_Reg_results()

@cli.command()
@click.option("test_url", "-t", default="https://www.gutenberg.org/files/98/98-0.txt")
@click.option('--algorithm',
              type=click.Choice(["skipgram", "cbow"], case_sensitive=False),
              default="skipgram")
def w2v(test_url, algorithm):
    local_file = tempfile.NamedTemporaryFile().name

    with urlopen(test_url) as response:
        resource = response.read()
        content = resource.decode('utf-8')
        fh = open(local_file, 'w')
        fh.write(content)

    encoder = xn2v.text_encoder.TextEncoder(local_file)
    data, count, dictionary, reverse_dictionary = encoder.build_dataset()
    print("Extracted a dataset with %d words" % len(data))
    if algorithm == 'cbow':
        logging.warning('Using cbow')
        model = xn2v.word2vec.ContinuousBagOfWordsWord2Vec(
                                data, worddictionary=dictionary,
                                reverse_worddictionary=reverse_dictionary)
    else:
        logging.warning('Using skipgram')
        model = SkipGramWord2Vec(data, worddictionary=dictionary,
                                 reverse_worddictionary=reverse_dictionary)
    model.add_display_words(count)
    model.train(display_step=1000)


if __name__ == "__main__":
    try:
        cli()
    except SystemExit as e:
        args = get_os_args()
        commands = list(cli.commands.keys())
        if args and args[0] not in commands:
            sys.exit("\nFirst argument should be a command: " + str(commands))
