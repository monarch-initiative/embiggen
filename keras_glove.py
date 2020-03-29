import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, Input, Add, Dot, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Model

import tarfile
from urllib.request import urlretrieve
import os
import nltk
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from scipy.sparse import save_npz, load_npz
from tensorflow.python.keras import backend as K
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import plot_model
import pandas as pd

url = 'http://www.cs.cmu.edu/~ark/personas/data/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    if not os.path.exists(os.path.join("datasets", filename)):
        print('Downloading file...')
        filename, _ = urlretrieve(url + filename, os.path.join("datasets", filename))
    else:
        print('File exists ...')

    print("Extracting the file")
    tar = tarfile.open(os.path.join("datasets", filename), "r:gz")
    tar.extractall("datasets")
    tar.close()

    statinfo = os.stat(os.path.join("datasets", filename))
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % os.path.join("datasets", filename))
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + os.path.join("datasets", filename) + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('MovieSummaries.tar.gz', 48002242)


def read_data(filename, n_lines):
    """ Reading the zip file to extract text """
    docs = []
    i = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for row in f:
            file_string = nltk.word_tokenize(row)
            # First token is the movie ID
            docs.append(' '.join(file_string[1:]))
            i += 1
            if n_lines and i == n_lines:
                break
    return docs


docs = read_data(os.path.join("datasets", "MovieSummaries", 'plot_summaries.txt'), 10000)
print("Read in {} documents".format(len(docs)))

v_size = 3000
tokenizer = Tokenizer(num_words=v_size, oov_token='UNK')
tokenizer.fit_on_texts(docs)

generate_cooc = False


def generate_cooc_matrix(text, tokenizer, window_size, n_vocab, use_weighting=True):
    sequences = tokenizer.texts_to_sequences(text)

    cooc_mat = lil_matrix((n_vocab, n_vocab), dtype=np.float32)
    for sequence in sequences:
        for i, wi in zip(np.arange(window_size, len(sequence) - window_size), sequence[window_size:-window_size]):
            context_window = sequence[i - window_size: i + window_size + 1]
            distances = np.abs(np.arange(-window_size, window_size + 1))
            distances[window_size] = 1.0
            nom = np.ones(shape=(window_size * 2 + 1,), dtype=np.float32)
            nom[window_size] = 0.0

            if use_weighting:
                cooc_mat[wi, context_window] += nom / distances  # Update element
            else:
                cooc_mat[wi, context_window] += nom
    return cooc_mat


cooc_file_path = os.path.join('datasets', 'cooc_mat.npz')

if generate_cooc or not os.path.exists(cooc_file_path):
    cooc_mat = generate_cooc_matrix(docs, tokenizer, 4, v_size, True)
    save_npz(cooc_file_path, cooc_mat.tocsr())
else:
    cooc_mat = load_npz(cooc_file_path).tolil()
    print('Cooc matrix of type {} was loaded from disk'.format(type(cooc_mat).__name__))

word = 'cat'
assert word in tokenizer.word_index, 'Word {} is not in the tokenizer'.format(word)
assert tokenizer.word_index[
           word] <= v_size, 'The word {} is an out of vocabuary word. Please try something else'.format(word)

rev_word_index = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))

cooc_vec = np.array(cooc_mat.getrow(tokenizer.word_index[word]).todense()).ravel()
max_ind = np.argsort(cooc_vec)[-25:]


# print(max_ind)


def plot_common_words():
    plt.figure(figsize=(16, 8))
    plt.bar(np.arange(0, 25), cooc_vec[max_ind])
    plt.xticks(ticks=np.arange(0, 25), labels=[rev_word_index[i] for i in max_ind], rotation=60)
    plt.show()


# plot_common_words()


def create_glove_model(v_size):
    """
    w_i = tf.keras.Input(shape=(1,), name="w_i") leads to a tensor with shape (None,1)
    emb_i = Flatten()(Embedding(v_size, 96, input_length=1)(w_i)) leads to a tensor with shape (None, v_size)
    both have dtype=float32
    :param v_size:
    :return:
    """
    w_i = tf.keras.Input(shape=(1,), name="w_i")
    w_j = tf.keras.Input(shape=(1,), name="w_j")

    emb_i = Flatten()(Embedding(v_size, 96, input_length=1)(w_i))
    emb_j = Flatten()(Embedding(v_size, 96, input_length=1)(w_j))

    ij_dot = Dot(axes=-1)([emb_i, emb_j])

    b_i = Flatten()(
        Embedding(v_size, 1, input_length=1)(w_i)
    )
    b_j = Flatten()(
        Embedding(v_size, 1, input_length=1)(w_j)
    )

    pred = Add()([ij_dot, b_i, b_j])

    def glove_loss(y_true, y_pred):
        return K.sum(
            K.pow((y_true - 1) / 100.0, 0.75) * K.square(y_pred - K.log(y_true))
        )

    model = Model(inputs=[w_i, w_j], outputs=pred)
    model.compile(loss=glove_loss, optimizer=Adam(lr=0.0001))
    return model

#K.clear_session()
model = create_glove_model(v_size)
model.summary()

cooc_mat = load_npz(os.path.join('datasets', 'cooc_mat.npz'))
batch_size = 128
copy_docs = list(docs)
index2word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
""" Each epoch """
for ep in range(10):

    # valid_words = get_valid_words(docs, 20, tokenizer)

    random.shuffle(copy_docs)
    losses = []
    """ Each document (i.e. movie plot) """
    for doc in copy_docs:

        seq = tokenizer.texts_to_sequences([doc])[0]

        """ Getting skip-gram data """
        # Negative samples are automatically sampled by tf loss function
        wpairs, labels = skipgrams(
            sequence=seq, vocabulary_size=v_size, negative_samples=0.0, shuffle=True
        )

        if len(wpairs) == 0:
            continue

        sg_in, sg_out = zip(*wpairs)
        sg_in, sg_out = np.array(sg_in).reshape(-1, 1), np.array(sg_out).reshape(-1, 1)
        x_ij = np.array(cooc_mat[sg_in[:, 0], sg_out[:, 0]]).reshape(-1, 1) + 1

        assert np.all(np.array(labels) == 1)
        assert x_ij.shape[0] == sg_in.shape[0], 'X_ij {} shape does not sg_in {}'.format(x_ij.shape, sg_in.shape)
        """ For each batch in the dataset """
        model.fit([sg_in, sg_out], x_ij, batch_size=batch_size, epochs=1, verbose=0)
        l = model.evaluate([sg_in, sg_out], x_ij, batch_size=batch_size, verbose=0)
        losses.append(l)
    print('Loss in epoch {}: {}'.format(ep, np.mean(losses)))