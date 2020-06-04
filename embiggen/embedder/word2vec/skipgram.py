class SkipGram(Word2Vec):
    """
    Class to run word2vec using skip grams
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
                tf.random.normal([self.vocabulary_size, self.embedding_size]))

            # construct the variables for the NCE loss
            self.nce_weights: tf.Variable = tf.Variable(
                tf.random.normal([self.vocabulary_size, self.embedding_size]))
            self.nce_biases: tf.Variable = tf.Variable(
                tf.zeros([self.vocabulary_size]))

    def fit(self, samples_per_window: int = 2):
        """Fit the Word2Vec skipgram model 
        Parameters
        ---------------------
        samples_per_window:
            samples_per_window: How many times to reuse an input to generate a label.
            
        """

        if samples_per_window > 2 * context_window:
            raise ValueError(
                'The value of self.samples_per_window must be <= twice the length of self.context_window')

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

            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=y,
                inputs=x_embed,
                number_negative_samples=self.number_negative_samples,
                num_classes=self.vocabulary_size
            ))

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
    #         x_embed_sqrt = tf.sqrt(tf.reduce_sum(tf.square(self._embedding), 1, keepdims=True), tf.float32)
    #         embedding_norm = self._embedding / x_embed_sqrt
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
                embedding = get_embedding(x, self._embedding, self.device_type)
                loss = self.nce_loss(embedding, y)

            # compute gradients
            gradients = g.gradient(
                loss, [self._embedding, self.nce_weights, self.nce_biases])

            # update W and b following gradients
            self.optimizer.apply_gradients(
                zip(gradients, [self._embedding, self.nce_weights, self.nce_biases]))

        return loss

    def next_batch(self, sentence: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training batch for the skip-gram model.

        Assumption: This assumes that dslist is a td.data.Dataset object that contains one sentence or (or list of words

        Args:
            sentence: A list of words to be used to create the batch
        Returns:
            A list where the first item is a batch and the second item is the batch's labels.
        Raises:
            ValueError: If the number of skips is not <= twice the skip window length.

        TODO -- should samples_per_window and context_window be arguments or simply taken from self
        within the method?
        """
        samples_per_window = self.samples_per_window
        context_window = self.context_window
        if samples_per_window > 2 * context_window:
            raise ValueError(
                'The value of self.samples_per_window must be <= twice the length of self.context_window')
        # TODO  -- We actually only need to check the above once in the Constructor?
        # OR -- is there any situation where we will change this during training??
        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * context_window + 1
        # again, probably we can go: span = self.span
        sentencelen = len(sentence)
        sentence = sentence.numpy()
        batch_size = ((sentencelen - (2 * context_window)) * samples_per_window)
        batch = np.empty(
            shape=(batch_size,),
            dtype=np.int32
        )
        labels = np.empty(
            shape=(batch_size, 1),
            dtype=np.int32)
        buffer: collections.deque = collections.deque(maxlen=span)
        # The following command fills up the Buffer but leaves out the last spot
        # this allows us to always add the next word as the first thing we do in the
        # following loop.
        buffer.extend(sentence[0:span - 1])
        data_index = span - 1
        for i in range(batch_size // samples_per_window):
            buffer.append(sentence[data_index])
            data_index += 1  # move sliding window 1 spot to the right
            context_words = [w for w in range(span) if w != context_window]
            words_to_use = random.sample(context_words, samples_per_window)

            for j, context_word in enumerate(words_to_use):
                batch[i * samples_per_window + j] = buffer[context_window]
                labels[i * samples_per_window + j, 0] = buffer[context_word]
        return batch, labels



    # TODO! this train method must receive the arguments that we don't need
    # TODO! most likely this method should be renames to 'fit'
    def train(self) -> List[float]:
        """
        Trying out passing a simple Tensor to get_batch
        :return:
        """
        window_len = 2 * self.context_window + 1
        step = 0
        loss_history = []
        for _ in trange(1, self.n_epochs + 1,  leave=False):
            if self.list_of_lists or isinstance(self.data, tf.RaggedTensor):
                for sentence in self.data:
                    # Sentence is a Tensor
                    sentencelen = len(sentence)
                    if sentencelen < window_len:
                        continue
                    batch_x, batch_y = self.next_batch(sentence)
                    # Evaluation.
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
                for _ in range(1, self.n_epochs + 1):
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
                        step += 1
                        self.run_optimization(batch_x, batch_y)
        return loss_history


