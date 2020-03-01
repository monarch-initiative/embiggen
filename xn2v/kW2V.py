############################
######
###### TEMPORARY FILE TO TEST KERAS-STYLE WORD2VEC CODE
######
############################

import random
import numpy as np   # type: ignore
import tensorflow as tf   # type: ignore
import collections

class NCE(tf.keras.layers.Layer):
    '''
    Custom Noise-Contrastive Estimation Loss
    '''
    def __init__(self, num_classes, neg_samples=100, **kwargs):

        self.num_classes = num_classes
        self.neg_samples = neg_samples

        super(NCE, self).__init__(**kwargs)

    # keras Layer interface
    def build(self, input_shape):

        self.W = self.add_weight(
            name="approx_softmax_weights",
            shape=(self.num_classes, input_shape[0][1]),
            initializer="glorot_normal",
        )

        self.b = self.add_weight(
            name="approx_softmax_biases", shape=(self.num_classes,), initializer="zeros"
        )

        # keras
        super(NCE, self).build(input_shape)

    # keras Layer interface
    def call(self, x):
        import tensorflow.keras.backend as K   # type: ignore
        predictions, targets = x

        # tensorflow
        loss = tf.nn.nce_loss(
            self.W, self.b, targets, predictions, self.neg_samples, self.num_classes
        )

        # keras
        self.add_loss(loss)

        logits = K.dot(predictions, K.transpose(self.W))

        return logits

    # keras Layer interface
    def compute_output_shape(self, input_shape):
        # return 1
        return self.num_classes

class kWord2Vec:
    """
    Superclass of all of the word2vec family algorithms.
    """

    def __init__(self,
                 data,
                 learning_rate=0.1,
                 batch_size=128,
                 num_steps=3000000,
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=1,  # default=2
                 skip_window=3,
                 num_skips=2,
                 num_sampled=7,  # default=64
                 display=None
                 ):
        """
        :param learning_rate:
        :param batch_size:
        :param num_steps: total iterations
        :param embedding_size: dimension of embedded vectors
        :param max_vocabulary_size: maximum number of words
        :param min_occurrence: minimum number of times a word needs to appear to be included
        :param skip_window: # How many words to consider left and right.
        :param num_skips: # How many times to reuse an input to generate a label.
        :param num_sampled: # Number of negative examples to sample.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.display_step = 2000
        self.eval_step = 2000
        self.embedding_dim = embedding_size
        self.max_vocabulary_size = max_vocabulary_size  # Total number of different words in the vocabulary.
        self.min_occurrence = min_occurrence
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.display = display
        self.display_examples = []
        self.data_index = 0
        self.current_sentence = 0

        # Q/C
        # 1. check if we have a flat list of ints or a list of lists of ints
        # 2. calculate vocabulary size
        if data is None:
            raise ValueError("input data cannot be null")
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) == 0:
            raise TypeError("data cannot be an empty list")
        first_elem = data[0]
        if isinstance(first_elem, list):
            self.list_of_lists = True
            if len(first_elem) < 0:
                raise TypeError("first list of data cannot be an empty list")
            flat_list = [item for sublist in data for item in sublist]
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(flat_list)))
            first_elem = first_elem[0]
        else:
            self.list_of_lists = False
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(data)))
            print("Vocabulary size (flat) is %d" % self.vocabulary_size)
        # when we get here, first elem must be a scalar integer of either the entire list or the first sublist
        if not isinstance(first_elem, int):
            raise TypeError("list elements must be integers")
        print("Vocabulary size: %d. listy of lists= %s" % (self.vocabulary_size, self.list_of_lists))
        self.max_vocabulary_size = min(50000, self.vocabulary_size)


        # self.embedding = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))
        # Construct the variables for the NCE loss.
        # self.nce_weights = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))
        # self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        with tf.device('/cpu:0'):
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Embedding(input_dim=self.vocabulary_size,  # embedding input
                                        output_dim=self.embedding_dim),   # embedding output
                tf.keras.layers.Dense(units=self.vocabulary_size,
                                    input_shape=(self.vocabulary_size, self.embedding_dim),
                                    activation=tf.nn.softmax,
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.RandomNormal())
            ])
            self.model.compile(loss=NCE(num_classes=self.vocabulary_size),
              optimizer=tf.optimizers.SGD(self.learning_rate),
              metrics=['accuracy'])

    def get_embedding(self, x):
        '''
        Get the embedding corresponding to the datapoints in x
        Note that we ensure that this code is carried out on the CPU because some ops are
        not compatible with thhe GPU
        :param x: data point indices, with shape (batch_size,)
        :return corresponding embeddings, with shape (batch_size, embedding_dimension)
        '''
        with tf.device('/cpu:0'):
            # Lookup the corresponding embedding vectors for each sample in X.
            x_embed = tf.nn.embedding_lookup(self.embedding, x)
            return x_embed
    def next_batch(self, data, batch_size, num_skips, skip_window):
        """
        Generate training batch for the skip-gram model. This assumes that all of the data is in one
        and only one list (for instance, the data might derive from a book). To get batches
        from a list of lists (e.g., node2vec), use the 'next_batch_from_list_of_list' function
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return:
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # get window size (words left and right + current one).
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(data):
            self.data_index = 0
        buffer.extend(data[self.data_index:self.data_index + span])
        # print("data", data[self.data_index:self.data_index + span])
        # print("bbuffer", buffer)
        self.data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                # j is the index of an element of words_to_use and context_word is that element
                # buffer -- the sliding window
                # buffer[skip_window] -- the center element of the sliding window
                #  buffer[context_word] -- the integer value of the word/node we are trying to predict with the skip-gram model
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(data):
                # i.e., we are at the end of data and need to reset the index to the beginning
                buffer.extend(data[0:span])
                self.data_index = span
            else:
                buffer.append(data[self.data_index])
                self.data_index += 1  # i.e., move the sliding window 1 position to the right
        # Backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels

    def next_batch_from_list_of_lists(self, walk_count, num_skips, skip_window):
        """
        Generate training batch for the skip-gram model. This assumes that all of the data is in one
        and only one list (for instance, the data might derive from a book). To get batches
        from a list of lists (e.g., node2vec), use the 'next_batch_from_list_of_list' function
        :param walk_count: number of walks (sublists or sentences) to ingest
        :param num_skips: The number of data points to extract for each center node
        :param skip_window: The size of the surrounding window (For instance, if skip_window=2 and num_skips=1,
        we look at 5 nodes at a time, and choose one data point from the 4 nodes that surround the center node
        :return: A batch of data points ready for learning
        """
        # assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * skip_window + 1
        batch = np.ndarray(shape=0, dtype=np.int32)
        labels = np.ndarray(shape=(0, 1), dtype=np.int32)
        for _ in range(walk_count):
            # here, sentence can be one random walk
            sentence = self.data[self.current_sentence]
            self.current_sentence += 1
            sentence_len = len(sentence)
            batch_count = sentence_len - span + 1
            current_batch, current_labels = self.next_batch(sentence, batch_count, num_skips, skip_window)
            batch = np.append(batch, current_batch)
            labels = np.append(labels, current_labels, axis=0)
            if self.current_sentence == self.num_sentences:
                self.current_sentence = 0
        return batch, labels

    def train(self, display_step=2000):
        # Words for testing.
        # Run training for the given number of steps.
        for step in range(1, self.num_steps + 1):
            if self.list_of_lists:
                walkcount = 2
                batch_x, batch_y = self.next_batch_from_list_of_lists(walkcount, self.num_skips, self.skip_window)
            else:
                batch_x, batch_y = self.next_batch(self.data, self.batch_size, self.num_skips, self.skip_window)
            self.model.fit(batch_x, batch_y, epochs=1)
            # possibly show progress??
            # self.model.evaluate() ?? Can we use this to show the nearest words
