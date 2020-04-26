import numpy as np
import csv
import collections
from scipy.sparse import lil_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

assert tf.__version__ >= "2.0"


def generate_batch_from_sentence(sentence, window_size):
    """Generate a batch of co-occurence counts for this sentence
  Args:
    sentence: a list of integers representing words or nodes
    window_size: the lenght of the window in which to count co-occurences
  Returns:
    batch, labels, weights
  """
    # two numpy arrays to hold target words (batch)
    # and context words (labels)
    # The weight is calculated to reflect the distance from the center word
    # This is the number of context words we sample for a single target word
    num_samples_per_centerword = 2 * window_size  # number of samples per center word
    num_contexts = len(sentence) - num_samples_per_centerword  # total n of full-length contexts in the sentence
    batch_size = num_contexts * num_samples_per_centerword  # For each context we sample num_samples time
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    weights = np.ndarray(shape=(batch_size), dtype=np.float32)
    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    span = 2 * window_size + 1
    # This is the number of context words we sample for a single target word
    num_samples_per_centerword = 2 * window_size
    buffer: collections.deque = collections.deque(maxlen=span)
    buffer.extend(sentence[0:span - 1])
    data_index = span - 1
    for i in range(num_contexts):
        buffer.append(sentence[data_index])
        data_index += 1  # move sliding window 1 spot to the right
        k = 0
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples_per_centerword + k] = buffer[window_size]
            labels[i * num_samples_per_centerword + k, 0] = buffer[j]
            weights[i * num_samples_per_centerword + k] = abs(1.0 / (j - window_size))
            k += 1
    return batch, labels, weights


def generate_cooc(sequences, vocab_size, window_size=2):
    """
    Generate co-occurence matrix by processing batches of data
    We are using the list of list sparse matrix from scipy
    """
    cooc_mat = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    print(cooc_mat.shape)
    i = 0
    for sequence in sequences:
        batch, labels, weights = generate_batch_from_sentence(sequence, window_size)
        labels = labels.reshape(-1)  # why is the reshape needed
        # Incrementing the sparse matrix entries accordingly
        for inp, lbl, w in zip(batch, labels, weights):
            cooc_mat[inp, lbl] += (1.0 * w)
        i += 1
        if i % 10==0:
            print("Sentence %d" % i)
    return cooc_mat


from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf


class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


class GloVeModelF:
    def __init__(self, co_oc_dict, vocab_size, embedding_size, context_size, min_occurrences=1,
                 scaling_factor=3 / 4, cooccurrence_cap=100, batch_size=128, learning_rate=0.05):
        self.co_oc_dict = co_oc_dict
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = 0.0001
        self.center_embeddings = tf.Variable(
            tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0), name='center.embeddings')
        self.center_bias = tf.Variable(tf.random.uniform([vocab_size], -1.0, 1.0, dtype=tf.float32),
                                       name='center.bias')
        self.context_embeddings = tf.Variable(
            tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0), name='context.embeddings')
        self.context_bias = tf.Variable(tf.random.uniform([vocab_size], -1.0, 1.0, dtype=tf.float32),
                                        name='context.bias')
        self.count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                     name='max_cooccurrence_count')
        self.scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                          name="scaling_factor")
        self.optimizer = tf.keras.optimizers.Adagrad(self.learning_rate)

    def get_embeds(self, train_dataset, train_labels):
        """
    @param: train_dataset -- a batch of center word indices
    @param: train_labels -- a batch of context word indices
    @return embeddings for center words and context words
    """
        with tf.device('cpu'):
            embed_in = tf.nn.embedding_lookup(self.center_embeddings, train_dataset)
            embed_out = tf.nn.embedding_lookup(self.context_embeddings, train_labels)
            embed_bias_in = tf.nn.embedding_lookup(self.center_bias, train_dataset)
            embed_bias_out = tf.nn.embedding_lookup(self.context_bias, train_labels)
            print("I AM IN get_embeds, and the type is ", type(embed_in) )
            return embed_in, embed_out, embed_bias_in, embed_bias_out

    def get_loss(self, weighting_factor, x_ij_s, embed_in, embed_out, embed_bias_in, embed_bias_out):
        """
    This calculates the loss according to the objective function of GloVe
    :param weighting_factor: GloVe weightings for current batch
    :param x_ij_s: co-occurences counts of words i and j
    :param embed_in: embedding vectors for the batch of center words
    :param embed_out: embedding vectors for the batch of context words
    :param embed_bias_in: bias for the center word
    :param embed_bias_out: bias for the context word
    :return: Loss for current batch
    """
        embedding_product = tf.reduce_sum(tf.multiply(embed_in, embed_out), 1)
        if not isinstance(x_ij_s, tuple):
            raise TypeError("x_ij_s needs to be a tuple")
        if len(x_ij_s) == 0:
            raise ValueError("x_ij_s is empty")
        log_cooccurrences = tf.math.log(tf.dtypes.cast(x_ij_s, tf.float32))
        distance_expr = tf.square(tf.add_n([
            embedding_product,
            embed_bias_in,
            embed_bias_out,
            tf.negative(log_cooccurrences)]))
        single_losses = tf.multiply(weighting_factor, distance_expr)
        return single_losses

    def run_optimization(self, train_dataset, train_labels, x_ij_s):
        """
    TODO -- Figure out how to put this on GPU if we have a GPU
    :param train_dataset:
    :param train_labels:
    :param weighting_factor:
    :param x_ij_s:
    :return: run optimization for one batch and return the loss
    """
        with tf.device('/cpu:0'):
            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.math.divide(x_ij_s, self.count_max),
                    self.scaling_factor))

            with tf.GradientTape() as g:
                embed_in, embed_out, embed_bias_in, embed_bias_out = self.get_embeds(train_dataset, train_labels)
                loss = self.get_loss(weighting_factor, x_ij_s, embed_in, embed_out, embed_bias_in, embed_bias_out)

            gradients = g.gradient(loss, [embed_in, embed_out, embed_bias_in, embed_bias_out])
            print("embed_in type:", type(embed_in))
            print("gradients type:", type(gradients))
            print("gradients len:", len(gradients))
            print("$$$$$$$$$$$$$$$$$   CRASH HAPPENS IN NEXT LINE")

            xx = zip(gradients, [embed_in, embed_out, embed_bias_in, embed_bias_out])
            print(xx)
            grads_and_vars = tuple(xx)
            if not grads_and_vars:
                print("NOT grads_and_vars")
            filtered = []
            vars_with_empty_grads = []
            for grad, var in grads_and_vars:
                if grad is None:
                    print("grad is none for var=", var)
                else:
                    filtered.append((grad, var))
            filtered = tuple(filtered)
            print("filtered len is", len(filtered))
            # print(filtered)
            self.optimizer.apply_gradients(zip(gradients, [embed_in, embed_out, embed_bias_in, embed_bias_out]))
            # self.optimizer.minimize  instead??
            print("NEVER GET HERE")
            return loss

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000):
        batches = self.__prepare_batches()
        total_steps = 0
        losses = []
        for epoch in range(num_epochs):
            shuffle(batches)
            for batch_index, batch in enumerate(batches):
                i_s, j_s, x_ij_s = batch
                if len(x_ij_s) != self.batch_size:
                    continue
                current_loss = self.run_optimization(i_s, j_s, x_ij_s)
                losses.append(current_loss)
                print("loss at epoch {}, batch index {}: {}".format(epoch, batch_index, current_loss))

    def _batchify(self, *sequences):
        for i in range(0, len(sequences[0]), self.batch_size):
            yield tuple(sequence[i:i + self.batch_size] for sequence in sequences)

    def __prepare_batches(self):
        if self.co_oc_dict is None:
            raise NotFitToCorpusError(
                "TOD) - REVISE NOT NEEDED Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.co_oc_dict.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(self._batchify(i_indices, j_indices, counts))


## For testing purposes, run it here

sentences = []
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]
with open("/home/peter/PycharmProjects/N2V/xn2v/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)
print("Number of sentences: %d" % len(sentences))
print(sentences[0])

vocab_size = 5000
oov_tok = '<OOV>'
max_length = 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
word_counts = tokenizer.word_counts
reverse_dictionary = dict(map(reversed, word_index.items()))
sequences = tokenizer.texts_to_sequences(sentences)

cooc_mat = generate_cooc(sequences[0:20], vocab_size, 2)
cooc_dict = cooc_mat.todok()

batch_size = 10
gf = GloVeModelF(co_oc_dict=cooc_dict, vocab_size=vocab_size, embedding_size=50, context_size=2)
gf.train(num_epochs=5)
