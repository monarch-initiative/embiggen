import collections
import math
import numpy as np
import random
import tensorflow as tf

from tqdm import trange
from typing import Dict, List, Optional, Tuple, Union

from xn2v import CBOWListBatcher
from xn2v.utils import get_embedding


class Word2Vec:
    """Superclass of all of the word2vec family algorithms.

        Attributes:
            learning_rate: A float between 0 and 1 that controls how fast the model learns to solve the problem.
            batch_size: The size of each "batch" or slice of the data to sample when training the model.
            num_steps: The number of epochs to run when training the model.
            display_step: An integer that is used to determine the number of steps to display.
            eval_step: This attribute stores the total number of iterations to run during training.
            embedding_size: Dimension of embedded vectors.
            max_vocabulary_size: Maximum number of words (i.e. total number of different words in the vocabulary).
            vocabulary_size: An integer storing the total number of unique words in the vocabulary.
            min_occurrence: Minimum number of times a word needs to appear to be included (default=2).
            skip_window: How many words to consider left and right.
            num_skips: How many times to reuse an input to generate a label.
            num_sampled: Number of negative examples to sample (default=64).
            display: An integer of the number of words to display.
        """

    def __init__(self, learning_rate: float = 0.1, batch_size: int = 128,
                 num_steps: int = 3000000, embedding_size: int = 200, max_vocabulary_size: int = 50000,
                 min_occurrence: int = 1, skip_window: int = 3, num_skips: int = 2, num_sampled: int = 7,
                 display: Optional[int] = None, device_type: str = 'cpu') -> None:

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.display_step = 2000
        self.eval_step = 2000
        self.embedding_size = embedding_size
        self.max_vocabulary_size = max_vocabulary_size
        self.vocabulary_size: int = 0
        self.min_occurrence = min_occurrence
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.display = display
        self.display_examples = []
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'

    def add_display_words(self, count: list, num: int = 5) -> None:
        """Creates a list of display nodes/words by obtaining a random sample of 'num' nodes/words from the full
        sample.

        If the argument 'display' is not None, then we expect that the user has passed an integer amount of words
        that are to be displayed together with their nearest neighbors, as defined by the word2vec algorithm. It is
        important to note that display is a costly operation. Up to 16 nodes/words are permitted. If a user asks for
        more than 16, a random validation set of 'num' nodes/words, that includes common and uncommon nodes/words, is
        selected from a 'valid_window' of 50 nodes/words.

        Args:
            count: A list of tuples (key:word, value:int).
            num: An integer representing the number of words to sample.

        Returns:
            None.

        Raises:
            TypeError: If the user does not provide a list of display words.
        """

        if not isinstance(count, list):
            self.display = None
            raise TypeError('self.display requires a list of tuples with key:word, value:int (count)')

        if num > 16:
            print('WARNING: maximum of 16 display words allowed (you passed {num_words})'.format(num_words=num))
            num = 16

        # pick a random validation set of 'num' words to sample
        valid_window = 50
        valid_examples = np.array(random.sample(range(2, valid_window), num))

        # sample less common words - choose 'num' points randomly from the first 'valid_window' after element 1000
        self.display_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), num), axis=0)

        return None

    def calculate_vocabulary_size(self, data) -> None:
        """Calculates the vocabulary size for the input data, which is a list of words (i.e. from a text),
        or list of lists (i.e. from a collection of sentences or random walks).

        Args:
            data: A list or list of lists (if sentences or paths from node2vec).

        Returns:
            None.
        """

        if any(isinstance(el, list) for el in data):
            flat_list = [item for sublist in data for item in sublist]
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(flat_list)) + 1)
            print('Vocabulary size (list of lists) is {vocab_size}'.format(vocab_size=self.vocabulary_size))
        else:
            self.vocabulary_size = min(self.max_vocabulary_size, len(set(data)) + 1)
            print('Vocabulary size (flat) is {vocab_size}'.format(vocab_size=self.vocabulary_size))

        return None


class SkipGramWord2Vec(Word2Vec):
    """
    Class to run word2vec using skip grams
    """

    def __init__(self, data: List, worddictionary: Dict[str, int], reverse_worddictionary: Dict[int, str],
                 learning_rate: float = 0.1, batch_size: int = 128, num_steps: int = 100, embedding_size: int = 200,
                 max_vocabulary_size: int = 50000, min_occurrence: int = 1, skip_window: int = 3, num_skips: int = 2,
                 num_sampled: int = 7, display: Optional[int] = None, device_type: str = 'cpu') -> None:

        super().__init__(learning_rate, batch_size, num_steps, embedding_size, max_vocabulary_size, min_occurrence,
                         skip_window, num_skips, num_sampled, display, device_type)
        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'

        # takes the input data and goes through each element
        # first, check each element is a list
        if any(isinstance(el, list) for el in self.data):
            for el in self.data:
                # then, check each element of the list is integer
                if any(isinstance(item, int) for item in el):
                    # graph version
                    self.list_of_lists: bool = True
                else:
                    self.list_of_lists = False
                    raise TypeError('self.data must contain a list of walks where each walk is a sequence of integers.')

        # set vocabulary size
        self.calculate_vocabulary_size(self.data)

        # with toy exs the # of nodes might be lower than the default value of num_sampled of 64. num_sampled needs to
        # be less than the # of exs (num_sampled is the # of negative samples that get evaluated per positive ex)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = int(self.vocabulary_size / 2)

        self.optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD(learning_rate)
        self.data_index: int = 0
        self.current_sentence: int = 0
        self.num_sentences: int = len(self.data)

        # do not display examples during training unless the user calls add_display_words (i.e. default is None)
        self.display = display

        # ensure the following ops & var are assigned on CPU (some ops are not compatible on GPU)
        with tf.device(self.device_type):
            # create embedding (each row is a word embedding vector) with shape (#n_words, dims) and dim = vector size
            self.embedding: tf.Variable = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))

            # construct the variables for the NCE loss
            self.nce_weights: tf.Variable = tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))
            self.nce_biases: tf.Variable = tf.Variable(tf.zeros([self.vocabulary_size]))

    def nce_loss(self, x_embed: tf.Tensor, y: np.ndarray) -> Union[float, int]:
        """Calculates the noise-contrastive estimation (NCE) training loss estimation for each batch.

        Args:
            x_embed: A Tensor with shape [batch_size, dim].
            y: An array containing the target classes with shape [batch_size, num_true].

        Returns:
            loss: The NCE losses.
        """

        with tf.device(self.device_type):
            y = tf.cast(y, tf.int64)

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                                 biases=self.nce_biases,
                                                 labels=y,
                                                 inputs=x_embed,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.vocabulary_size)
                                  )

            return loss

    def evaluate(self, x_embed: tf.Tensor) -> Union[np.ndarray, tf.Tensor]:
        """Computes the cosine similarity between a provided embedding and all other embedding vectors.

        Args:
            x_embed: A Tensor containing word embeddings.

        Returns:
            cosine_sim_op: A tensor of the cosine similarities between input data embedding and all other embeddings.
        """

        with tf.device(self.device_type):
            x_embed_cast = tf.cast(x_embed, tf.float32)
            x_embed_norm = x_embed_cast / tf.sqrt(tf.reduce_sum(tf.square(x_embed_cast)))
            x_embed_sqrt = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True), tf.float32)
            embedding_norm = self.embedding / x_embed_sqrt

            # calculate cosine similarity
            cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)

            return cosine_sim_op

    def next_batch(self, data: List[Union[int, str]], batch_size: int, num_skips: int, skip_window: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Generates a training batch for the skip-gram model.

        Assumptions: All of the data is in one and only one list (for instance, the data might derive from a book).

        Args:
            data: A list of words or nodes.
            batch_size: An integer specifying the size of the batch to generate.
            num_skips: The number of data points to extract for each center node.
            skip_window: The size of sampling windows (technically half-window). The window of a word `w_i` will be
                `[i - window_size, i + window_size+1]`.

        Returns:
            A list where the first item is a batch and the second item is the batch's labels.

        Raises:
            - ValueError: If the batch size is not evenly divisible by the number of skips.
            - ValueError: If the number of skips is not <= twice the skip window length.
        """

        # check that batch_size is evenly divisible by num_skips and num_skips is less or equal to skip window size
        if batch_size % num_skips != 0:
            raise ValueError('The value of self.batch_size must be evenly divisible by the value of self.num_skips')
        if num_skips > 2 * skip_window:
            raise ValueError('The value of self.num_skips must be <= twice the length of self.skip_window')

        batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # get window size (words left and right + current one)
        span = (2 * skip_window) + 1
        buffer: collections.deque = collections.deque(maxlen=span)

        if self.data_index + span > len(data):
            self.data_index = 0

        buffer.extend(data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]

            if self.data_index == len(data):
                # when the end of the string is reached, reset to beginning
                buffer.extend(data[0:span])
                self.data_index = span
            else:
                buffer.append(data[self.data_index])

                # move the sliding window 1 position to the right
                self.data_index += 1

        # backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(data) - span) % len(data)

        return batch, labels

    def next_batch_from_list_of_lists(self, walk_count: int, num_skips: int, skip_window: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Generate training batch for the skip-gram model.

        Assumption: This assumes that all of the data is stored as a list of lists (e.g., node2vec).

        Args:
            walk_count: The number of walks (sublists or sentences) to ingest.
            num_skips: The number of data points to extract for each center node.
            skip_window: The size of sampling windows (technically half-window). The window of a word `w_i` will be
                `[i - window_size, i + window_size+1]`.

        Returns:
            A list where the first item us a batch and the second item is the batch's labels.

        Raises:
           ValueError: If the number of skips is not <= twice the skip window length.
        """

        if num_skips > 2 * skip_window:
            raise ValueError('The value of self.num_skips must be <= twice the length of self.skip_window')

        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * skip_window + 1
        batch = np.ndarray(shape=(0,), dtype=np.int32)
        labels = np.ndarray(shape=(0, 1), dtype=np.int32)

        for i in range(walk_count):
            # sentence can be one random walk
            sentence = self.data[self.current_sentence]
            batch_count = (len(sentence) - span) + 1

            # get batch data
            current_batch, current_labels = self.next_batch(sentence, batch_count, num_skips, skip_window)
            batch = np.append(batch, current_batch)
            labels = np.append(labels, current_labels, axis=0)

            self.current_sentence += 1
            if self.current_sentence == self.num_sentences:
                self.current_sentence = 0

        return batch, labels

    def run_optimization(self, x: np.array, y: np.array) -> None:
        """Runs optimization for each batch by retrieving an embedding and calculating loss. Once the loss has been
        calculated, the gradients are computed and the weights and biases are updated accordingly.

        Args:
            x: An array of integers to use as batch training data.
            y: An array of labels to use when evaluating loss for an epoch.

        Returns:
            None.
        """

        with tf.device(self.device_type):
            # wrap computation inside a GradientTape for automatic differentiation
            with tf.GradientTape() as g:
                embedding = get_embedding(x, self.embedding, self.device_type)
                loss = self.nce_loss(embedding, y)

            # compute gradients
            gradients = g.gradient(loss, [self.embedding, self.nce_weights, self.nce_biases])

            # update W and b following gradients
            self.optimizer.apply_gradients(zip(gradients, [self.embedding, self.nce_weights, self.nce_biases]))

        return None

    def train(self, display_step: int = 2000) -> None:
        """Trains a SkipGram model.

        Args:
            display_step: An integer that is used to determine the number of steps to display when training the model.

        Returns:
            None.
        """

        # words for testing; display_step = 2000; eval_step = 2000
        do_display = self.display_step is not None and len(self.display_examples) > 0

        if do_display:
            for w in self.display_examples:
                print('{word}: id={index}'.format(word=self.id2word[w], index=w))

        x_test = np.array(self.display_examples)

        # run training for the given number of steps.
        with trange(1, self.num_steps + 1) as pbar:
            for step in pbar:
                if self.list_of_lists:
                    walkcount = 2
                    batch_x, batch_y = self.next_batch_from_list_of_lists(walkcount, self.num_skips, self.skip_window)
                else:
                    batch_x, batch_y = self.next_batch(self.data, self.batch_size, self.num_skips, self.skip_window)
                self.run_optimization(batch_x, batch_y)

                if step % display_step == 0 or step == 1:
                    loss = self.nce_loss(get_embedding(batch_x, self.embedding, self.device_type), batch_y)
                    pbar.set_description("step: %i, loss: %f" % (step, loss))

            # Evaluation.
            if do_display and (step % self.eval_step == 0 or step == 1):
                print("Evaluation...")
                sim = self.evaluate(get_embedding(x_test, self.embedding, self.device_type)).numpy()
                print(sim[0])
                for i in range(len(self.display_examples)):
                    top_k = 8  # number of nearest neighbors.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    disp_example = self.id2word[self.display_examples[i]]
                    log_str = '"%s" nearest neighbors:' % disp_example
                    for k in range(top_k):
                        log_str = '%s %s,' % (log_str, self.id2word[nearest[k]])
                    print(log_str)

        return None


class ContinuousBagOfWordsWord2Vec(Word2Vec):
    """Class to run word2vec using continuous bag of words (cbow).

    Attributes:
        word2id: A dictionary where the keys are nodes/words and values are integers that represent those nodes/words.
        id2word: A dictionary where the keys are integers and values are the nodes represented by the integers.
        data: A list or list of lists (if sentences or paths from node2vec).
        device_type: A string that indicates whether to run computations on (default=cpu).
        learning_rate: A float between 0 and 1 that controls how fast the model learns to solve the problem.
        batch_size: The size of each "batch" or slice of the data to sample when training the model.
        num_steps: The number of epochs to run when training the model (default=3000000).
        embedding_size: Dimension of embedded vectors.
        max_vocabulary_size: Maximum number of words (i.e. total number of different words in the vocabulary).
        min_occurrence: Minimum number of times a word needs to appear to be included (default=2).
        skip_window: How many words to consider left and right.
        num_skips: How many times to reuse an input to generate a label.
        num_sampled: Number of negative examples to sample (default=64).
        display: An integer of the number of words to display.
        embedding: A 2D tensor with shape (samples, sequence_length), where each entry is a sequence of integers.
        batcher: A list of CBOW data for training; the first item is a batch and the second item is the batch labels.
        list_of_lists: A boolean which indicates whether or not the input data contains a list of lists.
        optimizer: The TensorFlow optimizer to use.
        data_index: An integer that stores the index of data for use when creating batches.
        current_sentence: An integer which is used to track the number of sentences or random walks.
        num_sentences: An integer that stores the total number of sentences.
        softmax_weights: A variable that stores the classifier weights.
        softmax_biases: A variable that stores classifier biases.

    Raises:
        TypeError: If the self.data does not contain a list of lists, where each list contains integers.
    """

    def __init__(self, data: List, worddictionary: Dict[str, int], reverse_worddictionary: Dict[int, str],
                 learning_rate: float = 0.1, batch_size: int = 128, num_steps: int = 1000, embedding_size: int = 200,
                 max_vocabulary_size: int = 50000, min_occurrence: int = 1, skip_window: int = 3, num_skips: int = 2,
                 num_sampled: int = 7, display: Optional[int] = None, device_type: str = 'cpu') -> None:

        super().__init__(learning_rate, batch_size, num_steps, embedding_size, max_vocabulary_size, min_occurrence,
                         skip_window, num_skips, num_sampled, display, device_type)

        sentences_per_batch = 1

        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'
        self.batcher: CBOWListBatcher = CBOWListBatcher(data, skip_window, sentences_per_batch)

        # takes the input data and goes through each element
        # first, check each element is a list
        if any(isinstance(el, list) for el in self.data):
            for el in self.data:
                # then, check each element of the list is integer
                if any(isinstance(item, int) for item in el):
                    self.list_of_lists: bool = True
                else:
                    self.list_of_lists = False
                    raise TypeError('self.data must contain a list of walks where each walk is a sequence of integers.')

        # set vocabulary size
        self.calculate_vocabulary_size(self.data)

        # with toy exs the # of nodes might be lower than the default value of num_sampled of 64. num_sampled needs to
        # be less than the # of exs (num_sampled is the # of negative samples that get evaluated per positive ex)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = int(self.vocabulary_size / 2)

        self.optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD(learning_rate)
        self.data_index: int = 0
        self.current_sentence: int = 0
        self.num_sentences: int = len(self.data)

        # do not display examples during training unless the user calls add_display_words (i.e. default is None)
        self.display = display

        # ensure the following ops & var are assigned on CPU (some ops are not compatible on GPU)
        with tf.device(self.device_type):
            # create embedding (each row is a word embedding vector) with shape (#n_words, dims) and dim = vector size
            self.embedding: tf.Variable = tf.Variable(
                tf.random.uniform([self.vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32)
            )

            # should we initialize with uniform or normal?
            # # tf.Variable(tf.random.normal([self.vocabulary_size, embedding_size]))

            # construct the variables for the softmax loss
            tf_distribution = tf.random.truncated_normal([self.vocabulary_size, embedding_size],
                                                         stddev=0.5 / math.sqrt(embedding_size),
                                                         dtype=tf.float32)
            # get weights and biases
            self.softmax_weights: tf.Variable = tf.Variable(tf_distribution)
            self.softmax_biases = tf.Variable(tf.random.uniform([self.vocabulary_size], 0.0, 0.01))

    def cbow_embedding(self, x):
        """The function performs embedding lookups for each column in the input (except the middle one) and then
        averages the them to produce a word vector. The dimension of x is (batchsize, 2*skip_window), e.g., (128,6).

        Note. x does not contain the middle word.

        Args:
            x: A batch-size long list of windows of words (sliding windows), for example:
                [[ 2619 15572 15573 15575 15576 15577], [15572 15573 15574 15576 15577 15578], ...]

        Returns:
            mean_embeddings: An averaged word embedding vector.

        Raises:
            ValueError: If the shape of stacked_embeddings is not equal to twice the skip_window length.
        """

        stacked_embeddings = None
        # print('Defining %d embedding lookups representing each word in the context' % (2 * self.skip_window))
        for i in range(2 * self.skip_window):
            embedding_i = get_embedding(x[:, i], self.embedding)
            x_size, y_size = embedding_i.get_shape().as_list()  # added ',_' -- is this correct?

            if stacked_embeddings is None:
                stacked_embeddings = tf.reshape(embedding_i, [x_size, y_size, 1])
            else:
                stacked_embedding_value = [stacked_embeddings, tf.reshape(embedding_i, [x_size, y_size, 1])]
                stacked_embeddings = tf.concat(axis=2, values=stacked_embedding_value)

        assert stacked_embeddings.get_shape().as_list()[2] == 2 * self.skip_window
        mean_embeddings = tf.reduce_mean(stacked_embeddings, 2, keepdims=False)

        return mean_embeddings

    def get_loss(self, mean_embeddings: tf.Tensor, y: np.ndarray) -> Union[float, int]:
        """Computes the softmax loss, using a sample of the negative labels each time. The inputs are embeddings of the
        train words with this loss we optimize weights, biases, embeddings.

        Args:
            mean_embeddings: A Tensor with shape [batch_size, dim].
            y: An array containing the target classes with shape [batch_size, num_true].

        Returns:
            loss: The softmax losses.
        """

        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.softmax_weights,
                                                         biases=self.softmax_biases,
                                                         inputs=mean_embeddings,
                                                         labels=y,
                                                         num_sampled=self.num_sampled,
                                                         num_classes=self.vocabulary_size))

        return loss

    def nce_loss(self, x_embed: tf.Tensor, y: np.ndarray) -> Union[float, int]:
        """Calculates the noise-contrastive estimation (NCE) training loss estimation for each batch.

        Args:
            x_embed: A Tensor with shape [batch_size, dim].
            y: An array containing the target classes with shape [batch_size, num_true].

        Returns:
            loss: The NCE losses.
        """

        with tf.device(self.device_type):
            y = tf.cast(y, tf.int64)

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.softmax_weights,
                                                 biases=self.softmax_biases,
                                                 labels=y,
                                                 inputs=x_embed,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.vocabulary_size))

            return loss

    def evaluate(self, x_embed: tf.Tensor) -> tf.Tensor:
        """Computes the cosine similarity between a provided embedding and all other embedding vectors.

        Args:
            x_embed: A Tensor containing word embeddings.

        Returns:
            cosine_sim_op: A tensor of the cosine similarities between input data embedding and all other embeddings.
        """

        with tf.device(self.device_type):
            x_embed = tf.cast(x_embed, tf.float32)
            x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
            x_embed_sqrt = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True), tf.float32)
            embedding_norm = self.embedding / x_embed_sqrt

            # calculate cosine similarity
            cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)

            return cosine_sim_op

    def generate_batch_cbow(self, data: List[Union[int, str]], batch_size: int, window_size: int) ->\
            Tuple[np.ndarray, np.ndarray]:
        """Generates the next batch of data for CBOW.

        Args:
            data: A list of words or nodes. TODO make class variable
            batch_size:  An integer specifying the size of the batch to generate.
            window_size: The size of sampling windows (technically half-window). The window of a word `w_i` will be
                `[i - window_size, i + window_size+1]`.

        Returns:
            A list where the first item is a batch and the second item is the batch's labels.
        """

        # span is the total size of the sliding window we look at [skip_window central_word skip_window]
        span = 2 * window_size + 1

        # two numpy arrays to hold target (batch) and context words (labels). Batch has span-1=2*window_size columns
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int64)

        # The buffer holds the data contained within the span and the deque essentially implements a sliding window
        buffer: collections.deque = collections.deque(maxlen=span)

        # fill the buffer and update the data_index
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        # for each batch index, we iterate through span elements to fill in the columns of batch array
        for i in range(batch_size):
            target = window_size  # target word at the center of the buffer
            col_idx = 0

            for j in range(span):
                if j == span // 2:
                    continue  # i.e., ignore the center word
                batch[i, col_idx] = buffer[j]
                col_idx += 1

            labels[i, 0] = buffer[target]

            # move the span by 1, i.e., sliding window, since buffer is deque with limited size
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        return batch, labels

    def next_batch_from_list_of_lists(self, walk_count: int, num_skips: int, skip_window: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Generates training batch for the skip-gram model. This assumes that all of the data is in one and only one
        list (for instance, the data might derive from a book). To get batches from a list of lists (e.g., node2vec),
        use the 'next_batch_from_list_of_list' function.

        Args:
            walk_count: The number of walks (sublists or sentences) to ingest.
            num_skips: The number of data points to extract for each center node.
            skip_window: The size of sampling windows (technically half-window). The window of a word `w_i` will be
                `[i - window_size, i + window_size+1]`.

        Returns:
            A list where the first item us a batch and the second item is the batch's labels.

        Raises:
            ValueError: If the number of skips is not <= twice the skip window length.
        """

        if num_skips > 2 * skip_window:
            raise ValueError('The value of self.num_skips must be <= twice the self.skip_window length')

        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * skip_window + 1
        batch = np.ndarray(shape=(0,), dtype=np.int32)
        labels = np.ndarray(shape=(0, 1), dtype=np.int64)

        for _ in range(walk_count):
            sentence = self.data[self.current_sentence]
            self.current_sentence += 1
            sentence_len = len(sentence)
            batch_count = sentence_len - span + 1

            if self.list_of_lists:
                current_batch, current_labels = self.batcher.generate_batch()
                # self.next_batch_from_list_of_lists(sentence, batch_count, num_skips)
            else:
                current_batch, current_labels = self.generate_batch_cbow(sentence, batch_count, num_skips)

            batch = np.append(batch, current_batch)
            labels = np.append(labels, current_labels, axis=0)

            if self.current_sentence == self.num_sentences:
                self.current_sentence = 0

        return batch, labels

    def run_optimization(self, x: np.array, y: np.array) -> None:
        """Runs optimization for each batch by retrieving an embedding and calculating loss. Once the loss has
        been calculated, the gradients are computed and the weights and biases are updated accordingly.

        Args:
            x: An array of integers to use as batch training data.
            y: An array of labels to use when evaluating loss for an epoch.

        Returns:
            None.
        """

        with tf.device(self.device_type):
            # wrap computation inside a GradientTape for automatic differentiation
            with tf.GradientTape() as g:
                emb = self.cbow_embedding(x)
                loss = self.nce_loss(emb, y)

            # compute gradients
            gradients = g.gradient(loss, [self.embedding, self.softmax_weights, self.softmax_biases])

            # Update W and b following gradients
            self.optimizer.apply_gradients(
                zip(gradients, [self.embedding, self.softmax_weights, self.softmax_biases]))

            return None

    def train(self, display_step: int = 2000) -> None:
        """Trains a CBOW model.

        Args:
            display_step: An integer that is used to determine the number of steps to display when training the model.

        Returns:
            None.
        """

        # words for testing; display_step = 2000; eval_step = 2000
        if display_step is not None:
            for w in self.display_examples:
                print("{}: id={}".format(self.id2word[w], w))

        x_test = np.array(self.display_examples)

        # run training for the given number of steps.
        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = self.batcher.generate_batch()
            # self.generate_batch_cbow(self.data, self.batch_size, self.skip_window)

            self.run_optimization(batch_x, batch_y)

            if step % display_step == 0 or step == 1:
                loss = self.get_loss(self.cbow_embedding(batch_x), batch_y)
                print("step: %i, loss: %f" % (step, loss))

            # evaluation
            if self.display is not None and (step % self.eval_step == 0 or step == 1):
                print('Evaluation...\n')
                sim = self.evaluate(self.cbow_embedding(x_test)).numpy()
                print(sim[0])

                for i in range(len(self.display_examples)):
                    top_k = 8  # number of nearest neighbors.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    disp_example = self.id2word[self.display_examples[i]]
                    log_str = '{} nearest neighbors:'.format(disp_example)

                    for k in range(top_k):
                        log_str = '{} {},'.format(log_str, self.id2word[nearest[k]])

                    print(log_str)

            return None
