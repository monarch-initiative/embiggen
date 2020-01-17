import logging

import click
import n2v
from n2v import CSFGraph
from n2v.word2vec import SkipGramWord2Vec
from n2v import LinkPrediction

@click.group()
def cli():
    pass

@cli.command()
@click.option("training_file", "-t", type=click.Path(exists=True))
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

    hetgraph = n2v.hetnode2vec.N2vGraph(training_graph, p, q, gamma, use_gamma)
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
@click.option("test_file", "-t", type=click.Path(exists=True))
@click.option("training_file", "-r", type=click.Path(exists=True))
@click.option("embedded_graph", "-e", type=click.Path(exists=True))
def disease_link_prediction(test_file, training_file, embedded_graph):
    """
    Predict disease links
    """
    test_graph = CSFGraph(test_file)
    training_graph = CSFGraph(training_file)

    parameters = {'edge_embedding_method': "hadamard",
                  'portion_false_edges': 1}

    lp = LinkPrediction(training_graph, test_graph, embedded_graph, params=parameters)
    lp.predict_links()
    lp.output_Logistic_Reg_results()
    training_graph = CSFGraph(training_file)

if __name__ == "__main__":
    cli()
