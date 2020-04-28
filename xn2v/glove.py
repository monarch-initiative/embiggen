import numpy as np  # type: ignore
import csv


import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from random import shuffle

assert tf.__version__ >= "2.0"


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

        gradients = g.gradient(loss, [self.center_embeddings, self.context_embeddings, self.center_bias, self.context_bias])
        self.optimizer.apply_gradients(zip(gradients, [self.center_embeddings, self.context_embeddings, self.center_bias, self.context_bias]))
        # Note that the loss returned above is an array with the same dimension as batch size. For tracking the losss,
        # we will return the sum
        return tf.math.reduce_sum(loss)

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
                if batch_index % 50 == 0:
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

from xn2v import TextCooccurrenceEncoder
path = "/home/peter/data/n2v/treasure-island.txt"
with open(path, 'r') as file:
    tisland = file.read().replace('\n', '')
vocab_size=25000
tco = TextCooccurrenceEncoder(path, window_size=7, vocab_size=vocab_size)
cooc_dict, word_counts, word_index, reverse_dictionary  = tco.build_dataset()

batch_size = 10
gf = GloVeModelF(co_oc_dict=cooc_dict, vocab_size=vocab_size, embedding_size=50, context_size=5)
gf.train(num_epochs=5)
