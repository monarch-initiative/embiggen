from .word2vec import Word2Vec


class Cbow(Word2Vec):
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
        context_window: How many words to consider left and right.
        samples_per_window: How many times to reuse an input to generate a label.
        number_negative_samples: Number of negative examples to sample (default=7).
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

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # set vocabulary size
        self.calculate_vocabulary_size()

        # with toy exs the # of nodes might be lower than the default value of number_negative_samples of 7. number_negative_samples needs to
        # be less than the # of exs (number_negative_samples is the # of negative samples that get evaluated per positive ex)
        if self.number_negative_samples > self.vocabulary_size:
            self.number_negative_samples = int(self.vocabulary_size / 2)

        self.optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD(
            self.learning_rate)
        self.data_index: int = 0
        self.current_sentence: int = 0

        # ensure the following ops & var are assigned on CPU (some ops are not compatible on GPU)
        with tf.device(self.device_type):
            # create embedding (each row is a word embedding vector) with shape (#n_words, dims) and dim = vector size
            self._embedding: tf.Variable = tf.Variable(
                tf.random.uniform(
                    [self.vocabulary_size, self.embedding_size], -1.0, 1.0, dtype=tf.float32)
            )

            # should we initialize with uniform or normal?
            # # tf.Variable(tf.random.normal([self.vocabulary_size, self.embedding_size]))

            # construct the variables for the softmax loss
            tf_distribution = tf.random.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                         stddev=0.5 /
                                                                math.sqrt(
                                                                    self.embedding_size),
                                                         dtype=tf.float32)
            # get weights and biases
            self.softmax_weights: tf.Variable = tf.Variable(tf_distribution)
            self.softmax_biases = tf.Variable(
                tf.random.uniform([self.vocabulary_size], 0.0, 0.01))

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
            embedding_i = get_embedding(x[:, i], self._embedding)
            # added ',_' -- is this correct?
            x_size, y_size = embedding_i.get_shape().as_list()

            if stacked_embeddings is None:
                stacked_embeddings = tf.reshape(
                    embedding_i, [x_size, y_size, 1])
            else:
                stacked_embedding_value = [stacked_embeddings, tf.reshape(
                    embedding_i, [x_size, y_size, 1])]
                stacked_embeddings = tf.concat(
                    axis=2, values=stacked_embedding_value)

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
                                                         number_negative_samples=self.number_negative_samples,
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
                                                 number_negative_samples=self.number_negative_samples,
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
            x_embed_sqrt = tf.sqrt(tf.reduce_sum(
                tf.square(self._embedding), 1, keepdims=True), tf.float32)
            embedding_norm = self._embedding / x_embed_sqrt

            # calculate cosine similarity
            cosine_sim_op = tf.matmul(
                x_embed_norm, embedding_norm, transpose_b=True)

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
        sentencelen = sentence.shape[0]
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
                # TODO! This gets broken by ragged tensors!
                batch[i, j] = buffer[context_word]
            labels[i, 0] = buffer[context_window]
        return batch, labels

    def run_optimization(self, x: np.array, y: np.array):
        """Runs optimization for each batch by retrieving an embedding and calculating loss. Once the loss has
        been calculated, the gradients are computed and the weights and biases are updated accordingly.
        Args:
            x: An array of integers to use as batch training data.
            y: An array of labels to use when evaluating loss for an epoch.
        """
        with tf.device(self.device_type):
            # wrap computation inside a GradientTape for automatic differentiation
            with tf.GradientTape() as g:
                emb = self.cbow_embedding(x)
                loss = self.nce_loss(emb, y)

            # compute gradients
            gradients = g.gradient(
                loss, [self._embedding, self.softmax_weights, self.softmax_biases])

            # Update W and b following gradients
            self.optimizer.apply_gradients(
                zip(gradients, [self._embedding, self.softmax_weights, self.softmax_biases]))
            return tf.reduce_sum(loss)

    # TODO! this train method must receive the arguments that we don't need
    # TODO! most likely this method should be renames to 'fit'
    def train(self) -> List[None]:  # type: ignore
        """Trains a CBOW model.
        Returns:
            None.
        """
        window_len = 2 * self.context_window + 1
        step = 0
        loss_history = []
        for _ in trange(1, self.n_epochs + 1, leave=False):
            if self.list_of_lists or isinstance(self.data, tf.RaggedTensor):
                for sentence in self.data:
                    # Sentence is a Tensor
                    sentencelen = sentence.shape[0]
                    if sentencelen < window_len:
                        continue
                    batch_x, batch_y = self.generate_batch_cbow(sentence)
                    current_loss = self.run_optimization(
                        batch_x, batch_y)  # type: ignore
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
                    batch_x, batch_y = self.next_batch(
                        currentTensor)  # type: ignore
                    current_loss = self.run_optimization(
                        batch_x, batch_y)  # type: ignore
                    if step == 0 or step % 100 == 0:
                        logging.info("loss {}".format(current_loss))
                        loss_history.append(current_loss)
                    data_index += shift_len
                    endpos = data_index + batch_size
                    endpos = min(endpos,
                                 lastpos)  # takes care of last part of data. Maybe we should just ignore though
                    # Evaluation.
        return loss_history