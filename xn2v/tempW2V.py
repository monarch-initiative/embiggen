import tensorflow as tf  # type: ignore
import argparse
import os
import matplotlib.pyplot as plt  # type: ignore

from xn2v.text_encoder_tf import TextEncoder
from xn2v import SkipGramWord2Vec
from xn2v import ContinuousBagOfWordsWord2Vec
from xn2v.csf_graph.csf_graph import CSFGraph
from xn2v import N2vGraph

print(tf.__version__)
assert tf.__version__ >= "2.0"


parser = argparse.ArgumentParser(description='Run word2vec as test.')
parser.add_argument('-i', type=str, help='input file')
parser.add_argument('-w', type=bool,  nargs='?', const=True, default=False, help='do random walk')
parser.add_argument("-c", type=bool, nargs='?', const=True, default=False, help='do continuous bag of words')
args = parser.parse_args()
inputfile = args.i  # should be the path of a book from Project Gutenberg or a graph if run in graph mode
walk = args.w
cbow = args.c




def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()



def test_skipgram():
    encoder = TextEncoder(inputfile, data_type='words')
    tensor_data,  count_list, dictionary,  reverse_dictionary = encoder.build_dataset()
    print("count_list size=%d" % len(count_list))
    print("dictionary size=%d" % len(dictionary))
    print("reverse_dictionary size=%d" % len(reverse_dictionary))
    print(tensor_data)
    print(type(tensor_data))
    batch_size = 128
    window_shift=1
    sgw2v = SkipGramWord2Vec(data=tensor_data,
                         worddictionary=dictionary,
                         reverse_worddictionary=reverse_dictionary,
                         display=5)

    loss = sgw2v.train()
    plot_loss(loss)


def test_walk():
    print("walk")
    path = '../tests/data/ppt_train.txt'
    if not os.path.exists(path):
        raise ValueError("Could not find file for random walk")
    csf = CSFGraph(path)
    nodedictionary = csf.get_node_to_index_map()
    reverse_nodedictionary = csf.get_index_to_node_map()
    n2v = N2vGraph(csf, p=1, q=1, gamma=1, doxn2v=False)
    walks = n2v.simulate_walks(num_walks=10, walk_length=100)
    print(type(walks))
    print(walks[0:10])
    sgw2v = SkipGramWord2Vec(data=walks,
                             worddictionary=nodedictionary,
                             reverse_worddictionary=reverse_nodedictionary,
                             display=5)

    sgw2v.train()
    loss = sgw2v.train()
    print(loss[0:5])
    print("items in loss:",len(loss))
    plot_loss(loss)

def test_cbow():
    encoder = TextEncoder(inputfile, data_type='words')
    tensor_data, count_list, dictionary, reverse_dictionary = encoder.build_dataset()
    print("count_list size=%d" % len(count_list))
    print("dictionary size=%d" % len(dictionary))
    print("reverse_dictionary size=%d" % len(reverse_dictionary))
    print(tensor_data)
    print(type(tensor_data))
    batch_size = 128
    window_shift = 1
    sgw2v = ContinuousBagOfWordsWord2Vec(data=tensor_data,
                             worddictionary=dictionary,
                             reverse_worddictionary=reverse_dictionary,
                             display=5)

    loss = sgw2v.train()
    plot_loss(loss)

if cbow:
    test_cbow()
elif args.w is not None:
    test_walk()
else:
    test_skipgram()