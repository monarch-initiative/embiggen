import logging
import sys
import tempfile
from urllib.request import urlopen

import click as click
from click import get_os_args

import xn2v
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v import LinkPrediction
from xn2v.utils import write_embeddings

@click.group()
def cli():
    pass
#comment out hetrogenous graph embedding
# @cli.command()
# @click.option("training_file", "-t", type=click.Path(exists=True), required=True)
# @click.option("output_file", "-o", default='disease.embedded')
# @click.option("p", "-p", type=int, default=1)
# @click.option("q", "-q", type=int, default=1)
# @click.option("walk_length", "-w", type=int, default=80)
# @click.option("num_walks", "-n", type=int, default=25)
# @click.option("dimensions", "-d", type=int, default=128)
# @click.option("window_size", "-w", type=int, default=10)
# @click.option("workers", "-r", type=int, default=8)
# @click.option("num_epochs", "-s", type=int, default=1)
# def disease_gene_embeddings(training_file, output_file, p, q,
#                             walk_length, num_walks, dimensions, window_size, workers,
#                             num_epochs):
#     """
#     Generate disease gene embeddings
#     """
#     logging.basicConfig(level=logging.INFO)
#     print("Reading training file %s" % training_file)
#     training_graph = CSFGraph(training_file)
#     print(training_graph)
#     training_graph.print_edge_type_distribution()
#
#     hetgraph = xn2v.random_walk_generator.N2vGraph(training_graph, p, q)
#     walks = hetgraph.simulate_walks(num_walks, walk_length)
#     worddictionary = training_graph.get_node_to_index_map()
#     reverse_worddictionary = training_graph.get_index_to_node_map()
#
#
#     model = SkipGramWord2Vec(walks, worddictionary=worddictionary,
#                              reverse_worddictionary=reverse_worddictionary,
#                              num_epochs=num_epochs)
#     model.train()
#
#     write_embeddings(output_file, model.embedding, reverse_worddictionary)

@cli.command()

@click.option("positive_training_file", "-r", type=click.Path(exists=True), required=True)
@click.option("positive_test_file", "-t", type=click.Path(exists=True), required=True)
@click.option("negative_test_file",  type=click.Path(exists=True), required=True)
@click.option("negative_training_file", "-r", type=click.Path(exists=True), required=True)
@click.option("embedded_graph", "-e", type=click.Path(exists=True), required=True)
@click.option("edge_embedding_method", "-m", default="hadamard")
def disease_link_prediction(positive_training_file,
                            positive_test_file,
                            negative_training_file,
                            negative_test_file,
                            embedded_graph,
                            edge_embedding_method):
    """
    Predict disease links
    """

    training_graph = CSFGraph(positive_training_file)
    test_graph = CSFGraph(positive_test_file)
    negative_training_graph = CSFGraph(negative_training_file)
    negative_test_graph = CSFGraph(negative_test_file)
    lp = LinkPrediction(training_graph,
                        test_graph,
                        negative_training_graph,
                        negative_test_graph,
                        embedded_graph,
                        edge_embedding_method=edge_embedding_method)
    lp.predict_links()


@cli.command()
@click.option("training_file", "-t", type=click.Path(exists=True), required=True,
              default="tests/data/karate.train")
@click.option("test_file", "-t", type=click.Path(exists=True), required=True,
              default="tests/data/karate.test")
@click.option("output_file", "-o", default='karate.output')
@click.option("p", "-p", type=int, default=1)
@click.option("q", "-q", type=int, default=1)
@click.option("walk_length", "-w", type=int, default=80)
@click.option("num_epochs", "-n", type=int, default=1)
def karate_test(training_file, test_file, output_file, p, q,
                    walk_length, num_walks,num_epochs):
    training_graph = CSFGraph(training_file)
    graph = xn2v.random_walk_generator.N2vGraph(training_graph, p, q)

    walks = graph.simulate_walks(num_walks, walk_length)
    worddictionary = training_graph.get_node_to_index_map()
    reverse_worddictionary = training_graph.get_index_to_node_map()
    model = SkipGramWord2Vec(walks, worddictionary=worddictionary,
                             reverse_worddictionary=reverse_worddictionary,
                             num_epochs=num_epochs)
    model.train()
    output_filenname = 'karate.embedded'

    write_embeddings(output_filenname, model.embedding, reverse_worddictionary)

    test_graph = CSFGraph(test_file)
    path_to_embedded_graph = output_filenname


    lp = LinkPrediction(training_graph, test_graph, path_to_embedded_graph)#TODO:parameters of LinkPrediction

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
    model.train()


if __name__ == "__main__":
    try:
        cli()
    except SystemExit as e:
        args = get_os_args()
        commands = list(cli.commands.keys())
        if args and args[0] not in commands:
            sys.exit("\nFirst argument should be a command: " + str(commands))
