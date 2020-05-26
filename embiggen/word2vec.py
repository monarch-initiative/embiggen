import collections
import math
import numpy as np  # type: ignore
import random
import tensorflow as tf  # type: ignore

from tqdm import trange  # type: ignore
from typing import Dict, List, Optional, Tuple, Union

from embiggen.utils import get_embedding, calculate_cosine_similarity
import logging


class Word2Vec:
    """Superclass of all of the word2vec family algorithms.
        Attributes:
            learning_rate: A float between 0 and 1 that controls how fast the model learns to solve the problem.
            batch_size: The size of each "batch" or slice of the data to sample when training the model.
            num_epochs: The number of epochs to run when training the model.
            display_step: An integer that is used to determine the number of steps to display.
            eval_step: This attribute stores the total number of iterations to run during training.
            embedding_size: Dimension of embedded vectors.
            max_vocabulary_size: Maximum number of words (i.e. total number of different words in the vocabulary).
            vocabulary_size: An integer storing the total number of unique words in the vocabulary.
            min_occurrence: Minimum number of times a word needs to appear to be included (default=1).
            context_window: How many words to consider left and right.
            num_skips: How many times to reuse an input to generate a label.
            num_sampled: Number of negative examples to sample (default=7).
            display: An integer of the number of words to display.
        """

    def __init__(self, data: List, worddictionary: Dict[str, int], reverse_worddictionary: Dict[int, str],
                 learning_rate: float = 0.1, batch_size: int = 128,
                 num_epochs: int = 1, embedding_size: int = 200, max_vocabulary_size: int = 50000,
                 min_occurrence: int = 1, context_window: int = 3, num_skips: int = 2, num_sampled: int = 7,
                 display: Optional[int] = None, device_type: str = 'cpu') -> None:
        self.data = data
        self.word2id = worddictionary
        self.id2word = reverse_worddictionary
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = num_epochs
        self.display_step = 2000
        self.eval_step = 100
        self.embedding_size = embedding_size
        self.max_vocabulary_size = max_vocabulary_size
        self.vocabulary_size: int = 0
        self.min_occurrence = min_occurrence
        self.context_window = context_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.display = display
        self.display_examples: List = []
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'

        # takes the input data and goes through each element
        # first, check each element is a list
        # if any(isinstance(el, list) for el in self.data):
        if isinstance(self.data, tf.RaggedTensor):
            self.list_of_lists = True
        elif isinstance(self.data, tf.Tensor):
            self.list_of_lists = False
        else:
            logging.info("NEITHER RAGGED NOR TENSOR")
            logging.info("Type of data:{} ".format(type(self.data)))
            raise TypeError("NEITHER RAGGED NOR TENSOR")

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
            logging.warning('maximum of 16 display words allowed (you passed {num_words})'.format(num_words=num))
            num = 16

        # pick a random validation set of 'num' words to sample
        valid_window = 50
        valid_examples = np.array(random.sample(range(2, valid_window), num))

        # sample less common words - choose 'num' points randomly from the first 'valid_window' after element 1000
        self.display_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), num), axis=0)

        return None

    def calculate_vocabulary_size(self) -> None:
        """Calculates the vocabulary size for the input data, which is a list of words (i.e. from a text),
        or list of lists (i.e. from a collection of sentences or random walks).
        The function checks that self.vocabulary size has not been set

        Returns:
            None.
        """
        if self.word2id is None and self.vocabulary_size == 0:
            self.vocabulary_size = 0
        elif self.vocabulary_size ==0:
            self.vocabulary_size = len(self.word2id)+1
        # Apr 28, changed by Peter (can be deleted)
        # if any(isinstance(el, list) for el in data):
        #    flat_list = [item for sublist in data for item in sublist]
        #    self.vocabulary_size = min(self.max_vocabulary_size, len(set(flat_list)) + 1)
        #    print('Vocabulary size (list of lists) is {vocab_size}'.format(vocab_size=self.vocabulary_size))
        # else:
        ##    # was - self.vocabulary_size = min(self.max_vocabulary_size, len(set(data)) + 1)
        #    self.vocabulary_size = min(self.max_vocabulary_size, TFUtilities.gets_tensor_length(data) + 1)
        # print('Vocabulary size (flat) is {vocab_size}'.format(vocab_size=self.vocabulary_size))
        return None


class SkipGramWord2Vec(Word2Vec):
    """
    Class to run word2vec using skip grams
    """

    def __init__(self, data: List, worddictionary: Dict[str, int], reverse_worddictionary: Dict[int, str],
                 learning_rate: float = 0.1, batch_size: int = 128, num_epochs: int = 1, embedding_size: int = 200,
                 max_vocabulary_size: int = 50000, min_occurrence: int = 1, context_window: int = 3, num_skips: int = 2,
                 num_sampled: int = 7, display: Optional[int] = None, device_type: str = 'cpu') -> None:

        super().__init__(data=data, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary,
                         learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
                         embedding_size=embedding_size, max_vocabulary_size=max_vocabulary_size,
                         min_occurrence=min_occurrence, context_window=context_window, num_skips=num_skips,
                         num_sampled=num_sampled, display=display, device_type=device_type)

        self.data = data
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'
        # set vocabulary size
        self.calculate_vocabulary_size()

        # with toy exs the # of nodes might be lower than the default value of num_sampled of 7. num_sampled needs to
        # be less than the # of exs (num_sampled is the # of negative samples that get evaluated per positive ex)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = int(self.vocabulary_size / 2)

        self.optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD(learning_rate)
        self.data_index: int = 0
        self.current_sentence: int = 0

        # do not display examples during training unless the user calls add_display_words (i.e. default is None)
        self.display = display
        self.n_epochs = num_epochs

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

    # def evaluate(self, x_embed: tf.Tensor) -> Union[np.ndarray, tf.Tensor]:
    #     """Computes the cosine similarity between a provided embedding and all other embedding vectors.
    #
    #     Args:
    #         x_embed: A Tensor containing word embeddings.
    #
    #     Returns:
    #         cosine_sim_op: A tensor of the cosine similarities between input data embedding and all other embeddings.
    #     """
    #
    #     with tf.device(self.device_type):
    #         x_embed_cast = tf.cast(x_embed, tf.float32)
    #         x_embed_norm = x_embed_cast / tf.sqrt(tf.reduce_sum(tf.square(x_embed_cast)))
    #         x_embed_sqrt = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True), tf.float32)
    #         embedding_norm = self.embedding / x_embed_sqrt
    #
    #         # calculate cosine similarity
    #         cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
    #
    #         return cosine_sim_op

    def run_optimization(self, x: np.array, y: np.array) -> float:
        """Runs optimization for each batch by retrieving an embedding and calculating loss. Once the loss has been
        calculated, the gradients are computed and the weights and biases are updated accordingly.
        Args:
            x: An array of integers to use as batch training data.
            y: An array of labels to use when evaluating loss for an epoch.
        Returns:
            The loss of the current optimization round.
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

        return loss

    def next_batch(self, sentence: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training batch for the skip-gram model.

        Assumption: This assumes that dslist is a td.data.Dataset object that contains one sentence or (or list of words

        Args:
            sentence: A list of words to be used to create the batch
        Returns:
            A list where the first item us a batch and the second item is the batch's labels.
        Raises:
            ValueError: If the number of skips is not <= twice the skip window length.

        TODO -- should num_skips and context_window be arguments or simply taken from self
        within the method?
            """
        num_skips = self.num_skips
        context_window = self.context_window
        if num_skips > 2 * context_window:
            raise ValueError('The value of self.num_skips must be <= twice the length of self.context_window')
        # TODO  -- We actually only need to check the above once in the Constructor?
        # OR -- is there any situation where we will change this during training??
        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * context_window + 1
        # again, probably we can go: span = self.span
        sentencelen = len(sentence)
        batch_size = ((sentencelen - (2 * context_window)) * num_skips)
        batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        buffer: collections.deque = collections.deque(maxlen=span)
        sentence = sentence.numpy()
        # The following command fills up the Buffer but leaves out the last spot
        # this allows us to always add the next word as the first thing we do in the
        # following loop.
        buffer.extend(sentence[0:span - 1])
        data_index = span - 1
        for i in range(batch_size // num_skips):
            buffer.append(sentence[data_index])
            data_index += 1  # move sliding window 1 spot to the right
            context_words = [w for w in range(span) if w != context_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[context_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
        return batch, labels

    def display_words(self, x_test: np.array) -> None:
        """
        This is intended for to give feedback on the shell about the progress of training.
        It is not needed for the actual analysis.
        :param x_test:
        :return:
        """
        logging.info("Evaluation...")
        sim = calculate_cosine_similarity(get_embedding(x_test, self.embedding, self.device_type),
                                          self.embedding,
                                          self.device_type).numpy()
        #print(sim[0])
        for i in range(len(self.display_examples)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            disp_example = self.id2word[self.display_examples[i]]
            log_str = '"%s" nearest neighbors:' % disp_example
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, self.id2word[nearest[k]])
            #print(log_str)

    def train(self) -> List[float]:
        """
        Trying out passing a simple Tensor to get_batch
        :return:
        """
        # words for testing; display_step = 2000; eval_step = 2000
        do_display = self.display_step is not None and len(self.display_examples) > 0

        if do_display:
            for w in self.display_examples:
                logging.info('{word}: id={index}'.format(word=self.id2word[w], index=w))

        x_test = np.array(self.display_examples)

        window_len = 2 * self.context_window + 1
        step = 0
        loss_history = []
        for _ in trange(1, self.n_epochs + 1):
            if self.list_of_lists or isinstance(self.data, tf.RaggedTensor):
                for sentence in self.data:
                    # Sentence is a Tensor
                    sentencelen = len(sentence)
                    if sentencelen < window_len:
                        continue
                    batch_x, batch_y = self.next_batch(sentence)
                    # Evaluation.
                    if do_display and (step % self.eval_step == 0 or step == 0):
                        self.display_words(x_test)
                    step += 1
                    self.run_optimization(batch_x, batch_y)
            else:
                data = self.data  #
                if not isinstance(data, tf.Tensor):
                    raise TypeError("We were expecting a Tensor object!")
                # batch_size = self.batch_size
                data_len = len(data)
                # Note that we cannot fully digest all of the data in any one batch
                # if the window length is K and the natch_len is N, then the last
                # window that we get starts at position (N-K). Therefore, if we start
                # the next window at position (N-K)+1, we will get all windows.
                window_len = 1 + 2 * self.context_window
                shift_len = batch_size = window_len + 1
                # we need to make sure that we do not shift outside the boundaries of self.data too
                lastpos = data_len - 1  # index of the last word in data

                data_index = 0
                endpos = data_index + batch_size
                while True:
                    if endpos > lastpos:
                        break
                    currentTensor = data[data_index:endpos]
                    if len(currentTensor) < window_len:
                        break  # We are at the end
                    batch_x, batch_y = self.next_batch(currentTensor)
                    current_loss = self.run_optimization(batch_x, batch_y)
                    if step == 0 or step % 100 == 0:
                        logging.info("loss {}".format(current_loss))
                        loss_history.append(current_loss)
                    data_index += shift_len
                    endpos = data_index + batch_size
                    endpos = min(endpos,
                                 lastpos)  # takes care of last part of data. Maybe we should just ignore though
                    # Evaluation.
                    if do_display and (step % self.eval_step == 0 or step == 0):
                        self.display_words(x_test)
                    step += 1
                    self.run_optimization(batch_x, batch_y)
        return loss_history



class ContinuousBagOfWordsWord2Vec(Word2Vec):
    """Class to run word2vec using continuous bag of words (cbow).
    Attributes:
        word2id: A dictionary where the keys are nodes/words and values are integers that represent those nodes/words.
        id2word: A dictionary where the keys are integers and values are the nodes represented by the integers.
        data: A list or list of lists (if sentences or paths from node2vec).
        device_type: A string that indicates whether to run computations on (default=cpu).
        learning_rate: A float between 0 and 1 that controls how fast the model learns to solve the problem.
        batch_size: The size of each "batch" or slice of the data to sample when training the model.
        num_sepochs: The number of epochs to run when training the model (default=1).
        embedding_size: Dimension of embedded vectors.
        max_vocabulary_size: Maximum number of words (i.e. total number of different words in the vocabulary).
        min_occurrence: Minimum number of times a word needs to appear to be included (default=2).
        context_window: How many words to consider left and right.
        num_skips: How many times to reuse an input to generate a label.
        num_sampled: Number of negative examples to sample (default=7).
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
                 learning_rate: float = 0.1, batch_size: int = 128, num_epochs: int = 1, embedding_size: int = 200,
                 max_vocabulary_size: int = 50000, min_occurrence: int = 1, context_window: int = 3, num_skips: int = 2,
                 num_sampled: int = 7, display: Optional[int] = None, device_type: str = 'cpu') -> None:

        super().__init__(data=data, worddictionary=worddictionary, reverse_worddictionary=reverse_worddictionary,
                         learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
                         embedding_size=embedding_size, max_vocabulary_size=max_vocabulary_size,
                         min_occurrence=min_occurrence, context_window=context_window, num_skips=num_skips,
                         num_sampled=num_sampled, display=display, device_type=device_type)
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'
        # set vocabulary size
        self.calculate_vocabulary_size()

        # with toy exs the # of nodes might be lower than the default value of num_sampled of 7. num_sampled needs to
        # be less than the # of exs (num_sampled is the # of negative samples that get evaluated per positive ex)
        if self.num_sampled > self.vocabulary_size:
            self.num_sampled = int(self.vocabulary_size / 2)

        self.optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD(learning_rate)
        self.data_index: int = 0
        self.current_sentence: int = 0

        # do not display examples during training unless the user calls add_display_words (i.e. default is None)
        self.display = display
        self.n_epochs = num_epochs

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
        averages the them to produce a word vector. The dimension of x is (batchsize, 2*context_window), e.g., (128,6).
        Note. x does not contain the middle word.
        Args:
            x: A batch-size long list of windows of words (sliding windows), for example:
                [[ 2619 15572 15573 15575 15576 15577], [15572 15573 15574 15576 15577 15578], ...]
        Returns:
            mean_embeddings: An averaged word embedding vector.

        Raises:
            ValueError: If the shape of stacked_embeddings is not equal to twice the context_window length.
        """

        stacked_embeddings = None
        # print('Defining %d embedding lookups representing each word in the context' % (2 * self.context_window))
        for i in range(2 * self.context_window):
            embedding_i = get_embedding(x[:, i], self.embedding)
            x_size, y_size = embedding_i.get_shape().as_list()  # added ',_' -- is this correct?

            if stacked_embeddings is None:
                stacked_embeddings = tf.reshape(embedding_i, [x_size, y_size, 1])
            else:
                stacked_embedding_value = [stacked_embeddings, tf.reshape(embedding_i, [x_size, y_size, 1])]
                stacked_embeddings = tf.concat(axis=2, values=stacked_embedding_value)

        assert stacked_embeddings.get_shape().as_list()[2] == 2 * self.context_window
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

    def generate_batch_cbow(self, sentence: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the next batch of data for CBOW.
        Args:
            sentence: A list of words to be used to create the batch
        Returns:
            A list whose first item is a batch and the second item is the batch's labels.
        """
        # span is the total size of the sliding window we look at [context_window central_word context_window]
        context_window = self.context_window
        span = 2 * context_window + 1
        # again, probably we can go: span = self.span
        sentencelen = len(sentence)
        batch_size = (sentencelen - (2 * context_window))
        # two numpy arrays to hold target (batch) and context words (labels). Batch has span-1=2*window_size columns
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int64)
        sentence = sentence.numpy()
        # The buffer holds the data contained within the span and the deque essentially implements a sliding window
        buffer: collections.deque = collections.deque(maxlen=span)
        buffer.extend(sentence[0:span - 1])
        data_index = span - 1
        for i in range(batch_size):
            buffer.append(sentence[data_index])
            data_index += 1  # move sliding window 1 spot to the right
            context_words = [w for w in range(span) if w != context_window]
            for j, context_word in enumerate(context_words):
                batch[i, j] = buffer[context_word]
            labels[i, 0] = buffer[context_window]
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
            return tf.reduce_sum(loss)

    def display_words(self) -> None:
        for w in self.display_examples:
            logging.info("{}: id={}".format(self.id2word[w], w))
        # x_test = np.array(self.display_examples)
        logging.info('Evaluation...\n')
        print("TODO -- fix me")
        if 2 + 2 != 5:
            return
        sim = calculate_cosine_similarity(self.cbow_embedding(x_test),  # type: ignore
                                          self.embedding,
                                          self.device_type).numpy()

        #print(sim[0])

        for i in range(len(self.display_examples)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            disp_example = self.id2word[self.display_examples[i]]
            log_str = '{} nearest neighbors:'.format(disp_example)

            for k in range(top_k):
                log_str = '{} {},'.format(log_str, self.id2word[nearest[k]])

            #print(log_str)

        return None

    def train(self, display_step: int = 2000) -> List[None]:  # type: ignore
        """Trains a CBOW model.
        Args:
            display_step: An integer that is used to determine the number of steps to display when training the model.
        Returns:
            None.
        """
        # words for testing; display_step = 2000; eval_step = 2000
        if display_step is not None and 1 == 3:
            self.display_words()
        # data = self.data  #
        window_len = 2 * self.context_window + 1
        step = 0
        loss_history = []
        for _ in trange(1, self.n_epochs + 1):
            if self.list_of_lists or isinstance(self.data, tf.RaggedTensor):
                for sentence in self.data:
                    # Sentence is a Tensor
                    sentencelen = len(sentence)
                    if sentencelen < window_len:
                        continue
                    batch_x, batch_y = self.generate_batch_cbow(sentence)
                    current_loss = self.run_optimization(batch_x, batch_y)  # type: ignore
                    loss_history.append(current_loss)
                    if step % 100 == 0:
                        logging.info("loss {} ".format(current_loss))
                    step += 1
            else:
                data = self.data  #
                if not isinstance(data, tf.Tensor):
                    raise TypeError("We were expecting a Tensor object!")
                batch_size = self.batch_size
                data_len = len(data)
                # Note that we cannot fully digest all of the data in any one batch
                # if the window length is K and the natch_len is N, then the last
                # window that we get starts at position (N-K). Therefore, if we start
                # the next window at position (N-K)+1, we will get all windows.
                window_len = 1 + 2 * self.context_window
                shift_len = batch_size - window_len + 1
                # we need to make sure that we do not shift outside the boundaries of self.data too
                lastpos = data_len - 1  # index of the last word in data
                data_index = 0
                endpos = data_index + batch_size
                while endpos <= lastpos:
                    currentTensor = data[data_index:endpos]
                    if len(currentTensor) < window_len:
                        break  # We are at the end
                    batch_x, batch_y = self.next_batch(currentTensor)  # type: ignore
                    current_loss = self.run_optimization(batch_x, batch_y)  # type: ignore
                    if step == 0 or step % 100 == 0:
                        logging.info("loss {}".format(current_loss))
                        loss_history.append(current_loss)
                    data_index += shift_len
                    endpos = data_index + batch_size
                    endpos = min(endpos,
                                 lastpos)  # takes care of last part of data. Maybe we should just ignore though
                    # Evaluation.
            return loss_history
