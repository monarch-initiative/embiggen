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
                 min_occurrence=1,#default=2
                 skip_window = 3 ,
                 num_skips=2,
                 num_sampled=7,#default=64
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

    def write_embeddings(self, outfilename):
        if self.embedding is None:
            raise TypeError("Could not find self.embedding")
        if self.id2word is None:
            raise TypeError("Could not find self.id2word dictionary")
        id_list = self.id2word.keys()
        #id_list.sort() # TODO FIGURE OUT HOW TO SORT ME
        fh = open(outfilename, "w")
        with tf.device('/cpu:0'):
            for id in id_list:
                fh.write(self.id2word[id])
                x_embed = tf.nn.embedding_lookup(self.embedding, id)
                x = x_embed.numpy()
                for it in x:
                    fh.write("\t{}".format(it))
                fh.write("\n")




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
                 num_steps=1000,#default=3000000
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=1,#default=2
                 skip_window=3,
                 num_skips=2,
                 num_sampled=7,#default=64
                 display=None
                 ):
        super(SkipGramWord2Vec, self).__init__(learning_rate,
                                               batch_size,
                                               num_steps,
                                               embedding_size,
                                               max_vocabulary_size,
                                               min_occurrence,
                                               skip_window,
                                               num_skips,
                                               num_sampled,
                                               display)
        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        if any(isinstance(el, list) for el in self.data):
            self.list_of_lists = True
        else:
            self.list_of_lists = False
        self.calculate_vocabulary_size()
        # This should not be a problem with real data, but with toy examples the number of nodes might be
        # lower than the default value of num_sampled of 64. However, num_sampled needs to be less than
        # the number of examples (num_sampled is the number of negative samples that get evaluated per positive example)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = self.vocabulary_size/2
        self.optimizer = tf.optimizers.SGD(learning_rate)
        self.data_index = 0
        self.current_sentence = 0
        self.num_sentences = len(self.data)
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
            """
            print("self.nce_weights=%s (%s) " % (self.nce_weights, type(self.nce_weights)))
            print("self.nce_biases=%s (%s) " % (self.nce_biases,type(self.nce_biases)))
            print("y=%s (%s)" % (y,type(y)))
            print("x_embed=%s (%s) " % (x_embed,type(x_embed)))
            print("self.num_sampled=%s" % type(self.num_sampled))
            print("self.vocabulary_size=%s" % type(self.vocabulary_size))
            exit(1)
            """

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
                self.data_index += 1 # i.e., move the sliding window 1 position to the right
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
        for i in range(walk_count):
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

        # get window size (words left and right + current one).



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
            if self.list_of_lists:
                walkcount = 2
                batch_x, batch_y = self.next_batch_from_list_of_lists(walkcount, self.num_skips, self.skip_window)
            else:
                batch_x, batch_y = self.next_batch(self.data, self.batch_size, self.num_skips, self.skip_window)
            self.run_optimization(batch_x, batch_y)

            if step % display_step == 0 or step == 1:
                loss = self.nce_loss(self.get_embedding(batch_x), batch_y)
                print("step: %i, loss: %f" % (step, loss))

            # Evaluation.
            if not self.display is None and (step % self.eval_step == 0 or step == 1):
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

class ContinuousBagOfWordsWord2Vec(Word2Vec):
    """
    Class to run word2vec using skip grams
    """
    def __init__(self,
                 data,
                 worddictionary,
                 reverse_worddictionary,
                 learning_rate=0.1,
                 batch_size=128,
                 num_steps=1000,#default=3000000
                 embedding_size=200,
                 max_vocabulary_size=50000,
                 min_occurrence=1,#default=2
                 skip_window=3,
                 num_skips=2,
                 num_sampled=7,#default=64
                 display=None
                 ):
        super(SkipGramWord2Vec, self).__init__(learning_rate,
                                               batch_size,
                                               num_steps,
                                               embedding_size,
                                               max_vocabulary_size,
                                               min_occurrence,
                                               skip_window,
                                               num_skips,
                                               num_sampled,
                                               display)
        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        if any(isinstance(el, list) for el in self.data):
            self.list_of_lists = True
        else:
            self.list_of_lists = False
        self.calculate_vocabulary_size()
        # This should not be a problem with real data, but with toy examples the number of nodes might be
        # lower than the default value of num_sampled of 64. However, num_sampled needs to be less than
        # the number of examples (num_sampled is the number of negative samples that get evaluated per positive example)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = self.vocabulary_size/2
        self.optimizer = tf.optimizers.SGD(learning_rate)
        self.data_index = 0
        self.current_sentence = 0
        self.num_sentences = len(self.data)
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
            """
            print("self.nce_weights=%s (%s) " % (self.nce_weights, type(self.nce_weights)))
            print("self.nce_biases=%s (%s) " % (self.nce_biases,type(self.nce_biases)))
            print("y=%s (%s)" % (y,type(y)))
            print("x_embed=%s (%s) " % (x_embed,type(x_embed)))
            print("self.num_sampled=%s" % type(self.num_sampled))
            print("self.vocabulary_size=%s" % type(self.vocabulary_size))
            exit(1)
            """

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
                self.data_index += 1 # i.e., move the sliding window 1 position to the right
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
        for i in range(walk_count):
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

        # get window size (words left and right + current one).



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
            if self.list_of_lists:
                walkcount = 2
                batch_x, batch_y = self.next_batch_from_list_of_lists(walkcount, self.num_skips, self.skip_window)
            else:
                batch_x, batch_y = self.next_batch(self.data, self.batch_size, self.num_skips, self.skip_window)
            self.run_optimization(batch_x, batch_y)

            if step % display_step == 0 or step == 1:
                loss = self.nce_loss(self.get_embedding(batch_x), batch_y)
                print("step: %i, loss: %f" % (step, loss))

            # Evaluation.
            if not self.display is None and (step % self.eval_step == 0 or step == 1):
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
