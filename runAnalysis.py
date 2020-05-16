import logging
import sys
import tempfile
from urllib.request import urlopen

import click as click
from click import get_os_args

import embiggen
from embiggen import CSFGraph
from embiggen.word2vec import SkipGramWord2Vec
from embiggen import LinkPrediction
from embiggen.utils import write_embeddings

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
#     hetgraph = embiggen.random_walk_generator.N2vGraph(training_graph, p, q)
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

# @cli.command()
# @click.option("positive_training_file", "-r", type=click.Path(exists=True), required=True)
# @click.option("positive_test_file", "-t", type=click.Path(exists=True), required=True)
# @click.option("negative_test_file",  type=click.Path(exists=True), required=True)
# @click.option("negative_training_file", "-r", type=click.Path(exists=True), required=True)
# @click.option("embedded_graph", "-e", type=click.Path(exists=True), required=True)
# @click.option("edge_embedding_method", "-m", default="hadamard")
# def disease_link_prediction(positive_training_file,
#                             positive_test_file,
#                             negative_training_file,
#                             negative_test_file,
#                             embedded_graph,
#                             edge_embedding_method):
#     """
#     Predict disease links
#     """
#
#     training_graph = CSFGraph(positive_training_file)
#     test_graph = CSFGraph(positive_test_file)
#     negative_training_graph = CSFGraph(negative_training_file)
#     negative_test_graph = CSFGraph(negative_test_file)
#     lp = LinkPrediction(training_graph,
#                         test_graph,
#                         negative_training_graph,
#                         negative_test_graph,
#                         embedded_graph,
#                         edge_embedding_method=edge_embedding_method)
#     lp.predict_links()


@cli.command()
@click.option("pos_train_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/pos_train_edges')
@click.option("pos_valid_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/pos_validation_edges')
@click.option("pos_test_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/pos_test_edges')
@click.option("neg_train_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/neg_train_edges')
@click.option("neg_valid_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/neg_validation_edges')
@click.option("neg_test_file", "-t", type=click.Path(exists=True), required=True,
              default='tests/data/karate/neg_test_edges')
@click.option("output_file", "-o", default='pos_train_karate.embedded')
@click.option("p", "-p", type=int, default=1)
@click.option("q", "-q", type=int, default=1)
@click.option("walk_length", "-w", type=int, default=80)
@click.option("num_walks", "-nw", type=int, default=10)
@click.option("num_epochs", "-n", type=int, default=1)
@click.option("classifier", "-classifier", default='LR')
@click.option("edge_embed_method", "-edge_embed_method", default='hadamard')
@click.option("useValidation", "-useValidation", default='True')

def karate_test(pos_train_file, pos_valid_file, pos_test_file, neg_train_file, neg_valid_file, neg_test_file, output_file, p, q,
                    walk_length, num_walks,num_epochs, classifier, edge_embed_method, useValidation):
    pos_train_graph = CSFGraph(pos_train_file)
    pos_valid_graph = CSFGraph(pos_valid_file)
    pos_test_graph = CSFGraph(pos_test_file)
    neg_train_graph = CSFGraph(neg_train_file)
    neg_valid_graph = CSFGraph(neg_valid_file)
    neg_test_graph = CSFGraph(neg_test_file)
    # Graph (node) embeding using SkipGram as the word2vec model, with 2 epochs.
    graph = embiggen.random_walk_generator.N2vGraph(pos_train_graph, p, q)
    walks = graph.simulate_walks(num_walks, walk_length)
    worddictionary = pos_train_graph.get_node_to_index_map()
    reverse_worddictionary = pos_train_graph.get_index_to_node_map()
    model = SkipGramWord2Vec(walks, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary,
                             num_epochs=num_epochs)
    model.train()
    write_embeddings(output_file, model.embedding, reverse_worddictionary)

    # Link prediction on the pos/neg train/valid/test sets using RF classifier
    lp = LinkPrediction(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph,
                        neg_test_graph,
                        output_file, edge_embed_method, classifier, useValidation)
    lp.prepare_edge_and_node_labels()
    lp.predict_links()
    lp.output_classifier_results()

@cli.command()
@click.option("test_url", "-t", default="https://www.gutenberg.org/files/98/98-0.txt")
@click.option('--algorithm',
              type=click.Choice(["skipgram", "cbow"], case_sensitive=False),
              default="skipgram")
@click.option("num_epochs", "-n", type=int, default=1)
@click.option("output_file", "-o", default='book.embedded')


def w2v(test_url, algorithm, num_epochs,output_file):
    local_file = tempfile.NamedTemporaryFile().name

    with urlopen(test_url) as response:
        resource = response.read()
        content = resource.decode('utf-8')
        fh = open(local_file, 'w')
        fh.write(content)

    encoder = embiggen.text_encoder.TextEncoder(local_file)
    data, count, dictionary, reverse_dictionary = encoder.build_dataset()
    #print("Extracted a dataset with %d words" % len(data))
    if algorithm == 'cbow':
        logging.warning('Using cbow')
        model = embiggen.word2vec.ContinuousBagOfWordsWord2Vec(
                                data, worddictionary=dictionary,
                                reverse_worddictionary=reverse_dictionary, num_epochs=num_epochs)
    else:
        logging.warning('Using skipgram')
        model = SkipGramWord2Vec(data, worddictionary=dictionary,
                                 reverse_worddictionary=reverse_dictionary, num_epochs=num_epochs)
    model.add_display_words(count)
    model.train()
    write_embeddings(output_file, model.embedding, reverse_dictionary)



if __name__ == "__main__":
    try:
        cli()
    except SystemExit as e:
        args = get_os_args()
        commands = list(cli.commands.keys())
        if args and args[0] not in commands:
            sys.exit("\nFirst argument should be a command: " + str(commands))
