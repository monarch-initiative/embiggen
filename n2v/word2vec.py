import random
import math
import numpy as np
import tensorflow as tf
import collections



class Word2Vec:
    """
    Superclass of all of the word2vec family algorithms.
    """
    def __init__(self,
                 learning_rate = 0.1,
                 batch_size = 128,
                 num_steps = 3000000,
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=2,
                 skip_window = 3 ,
                 num_skips=2,
                 num_sampled=64,
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
        self.embedding_size = embedding_size
        self.max_vocabulary_size = max_vocabulary_size  # Total number of different words in the vocabulary.
        self.min_occurrence = min_occurrence
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.display = display
        self.display_examples = []
        # Evaluation Parameters.
        # eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']
        #TODO add FUNCTION FOR EVALUATION
        eval_words = ['house', 'oliver', 'twist', 'nose']

    def add_display_words(self, count, num=5):
        '''
        If the argument 'display' is not None, then we expect that the user has passed
        an integer amount of words that are to be displayed together with their nearest
        neighbors as defined by the word2vec algorithm. This function finds the display
        words
        '''
        if not isinstance(count, list):
            self.display = None
            print("[WARNING] add_display_words requires a list of tuples with k:word v:int (count)")
            return
        if num > 16:
            print("[WARNING] maximum of 16 display words allowed (you passed %d)" % num)
            num = 16 # display is a costly operation, do not allow more than 10 words
        # Pick a random validation set of 'num' words to sample nearest neighbors
        # We sample 'num'' datapoints randomly from the first 'valid_window' elements
        valid_window = 50
        valid_examples = np.array(random.sample(range(valid_window), num))
        # We sample 'num'' datapoints randomly from the first 'valid_window' elements after element 1000
        # This is to sample some less common words
        self.display_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), num), axis=0)

    def calculate_vocabulary_size(self):
        """
        Calculate the vocabulary size
        """
        # self.data is either a list (e.g., from a text) or a list of lists (e.g., from a collection of random walks)
        if any(isinstance(el, list) for el in self.data):
            flat_list = [item for sublist in self.data for item in sublist]
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(flat_list)))
            print("Vocabulary size (list of lists) is %d" % self.vocabulary_size)
        else:
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(self.data)))
            print("Vocabulary size (flat) is %d" % self.vocabulary_size)



class SkipGramWord2Vec(Word2Vec):
    """
    Class to run word2vec using skip grams
    """

    def __init__(self,
                 data,
                 worddictionary,
                 reverse_worddictionary,
                 learning_rate=0.1,
                 batch_size=128,
                 num_steps=3000000,
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=2,
                 skip_window=3,
                 num_skips=2,
                 num_sampled=64
                 ):
        super(SkipGramWord2Vec, self).__init__(learning_rate,
                                               batch_size,
                                               num_steps,
                                               embedding_size,
                                               max_vocabulary_size,
                                               min_occurrence,
                                               skip_window,
                                               num_skips,
                                               num_sampled)
        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        self.calculate_vocabulary_size()
        self.optimizer = tf.optimizers.SGD(learning_rate)
        self.data_index = 0
        # Do not display examples during training unless the user calls add_display_words, i.e., default is None
        self.display = None
        # Ensure the following ops & var are assigned on CPU
        # (some ops are not compatible on GPU).
        with tf.device('/cpu:0'):
            # Create the embedding variable (each row represent a word embedding vector).
            self.embedding = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))
            # Construct the variables for the NCE loss.
            self.nce_weights = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))
            self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

    def get_embedding(self, x):
        # Ensure the following ops& var are assigned on CPU
        # (some ops are not compatible on GPU).
        with tf.device('/cpu:0'):
            # Lookup the corresponding embedding vectors for each sample in X.
            x_embed = tf.nn.embedding_lookup(self.embedding, x)
            return x_embed

    def nce_loss(self, x_embed, y):
        with tf.device('/cpu:0'):
            # Compute the average NCE loss for the batch.
            y = tf.cast(y, tf.int64)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=y,
                               inputs=x_embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocabulary_size))
            return loss

    # Evaluation.
    def evaluate(self, x_embed):
        with tf.device('/cpu:0'):
            # Compute the cosine similarity between input data embedding and every embedding vectors
            x_embed = tf.cast(x_embed, tf.float32)
            x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
            embedding_norm = self.embedding / tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True), tf.float32)
            cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
            return cosine_sim_op

    def next_batch(self, batch_size, num_skips, skip_window):
        """
        Generate training batch for the skip-gram model.
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return:
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        data = self.data
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
                # print("j: {} type {}".format(j, type(j)))
                # print("i: {} type {}".format(j, type(i)))
                # print("num_skips: {} type {}".format(num_skips, type(num_skips)))
                # print("skip_window: {} type {}".format(skip_window, type(skip_window)))
                # print("batch: {} type {}".format(batch[0], type(batch)))
                # print("buffer: {} type {}".format(buffer[0], type(buffer)))
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(data):
                buffer.extend(data[0:span])
                self.data_index = span
            else:
                buffer.append(data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels

    # Optimization process.
    def run_optimization(self, x, y):
        with tf.device('/cpu:0'):
            # Wrap computation inside a GradientTape for automatic differentiation.
            with tf.GradientTape() as g:
                emb = self.get_embedding(x)
                loss = self.nce_loss(emb, y)

            # Compute gradients.
            gradients = g.gradient(loss, [self.embedding, self.nce_weights, self.nce_biases])

            # Update W and b following gradients.
            self.optimizer.apply_gradients(zip(gradients, [self.embedding, self.nce_weights, self.nce_biases]))

    def train(self, display_step=2000):
        # Words for testing.
        #display_step = 2000
        #eval_step = 2000
        if display_step is not None:
            for w in self.display_examples:
                print("{}: id={}".format(self.id2word[w], w))

        x_test = np.array(self.display_examples)

        # Run training for the given number of steps.
        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = self.next_batch(self.batch_size, self.num_skips, self.skip_window)
            self.run_optimization(batch_x, batch_y)

            if step % display_step == 0 or step == 1:
                loss = self.nce_loss(self.get_embedding(batch_x), batch_y)
                print("step: %i, loss: %f" % (step, loss))

            # Evaluation.
            if step % self.eval_step == 0 or step == 1:
                print("Evaluation...")
                sim = self.evaluate(self.get_embedding(x_test)).numpy()
                print(sim[0])
                for i in range(len(self.display_examples)):
                    top_k = 8  # number of nearest neighbors.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    disp_example = self.id2word[self.display_examples[i]]
                    log_str = '"%s" nearest neighbors:' %  disp_example
                    for k in range(top_k):
                        log_str = '%s %s,' % (log_str, self.id2word[nearest[k]])
                    print(log_str)
