import tensorflow as tf
import numpy as np
import collections
import random
from typing import Union, Tuple, Dict, List
from tqdm.auto import trange
from .word2vec import Word2Vec


class SkipGram(Word2Vec):
    """
    Class to run word2vec using skip grams
    """

    def __init__(self) -> None:
        super().__init__()
        self.data_index: int = 0
        self.current_sentence: int = 0

    def nce_loss(self, x_embed: tf.Tensor, y: np.ndarray) -> Union[float, int]:
        """Calculates the noise-contrastive estimation (NCE) training loss estimation for each batch.
        Args:
            x_embed: A Tensor with shape [batch_size, dim].
            y: An array containing the target classes with shape [batch_size, num_true].
        Returns:
            loss: The NCE losses.
        """
        y = tf.cast(y, tf.int64)

        return tf.reduce_mean(tf.nn.nce_loss(
            weights=self._nce_weights,
            biases=self._nce_biases,
            labels=y,
            inputs=x_embed,
            num_sampled=self.number_negative_samples,
            num_classes=self._vocabulary_size
        ))

    def run_optimization(self, x: np.array, y: np.array) -> float:
        """Runs optimization for each batch by retrieving an embedding and calculating loss. Once the loss has been
        calculated, the gradients are computed and the weights and biases are updated accordingly.
        Args:
            x: An array of integers to use as batch training data.
            y: An array of labels to use when evaluating loss for an epoch.
        Returns:
            The loss of the current optimization round.
        """
        # wrap computation inside a GradientTape for automatic differentiation
        with tf.GradientTape() as g:
            with tf.device("cpu"):
                embedding = tf.nn.embedding_lookup(self._embedding, x)
            loss = self.nce_loss(embedding, y)

        # compute gradients
        gradients = g.gradient(
            loss, [self._embedding, self._nce_weights, self._nce_biases])

        # update W and b following gradients
        self.optimizer.apply_gradients(
            zip(gradients, [self._embedding, self._nce_weights, self._nce_biases]))

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
        
        # OR -- is there any situation where we will change this during training??
        # self.data is a list of lists, e.g., [[1, 2, 3], [5, 6, 7]]
        span = 2 * context_window + 1
        # again, probably we can go: span = self.span
        sentencelen = len(sentence)
        sentence = sentence.numpy()
        batch_size = ((sentencelen - (2 * context_window))
                      * samples_per_window)
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

    def _fit_list_of_lists(self, X: tf.RaggedTensor):
        window_len = 2 * self.context_window + 1
        batchnum = 0
        for epoch in trange(self.epochs):
            for sentence in X:
                # Sentence is a Tensor
                sentencelen = len(sentence)
                if sentencelen < window_len:
                    continue
                batch_x, batch_y = self.next_batch(sentence)
                batchnum += 1
                current_loss = self.run_optimization(batch_x, batch_y)
                self.on_batch_end(batch=batchnum, epoch=epoch, log={
                                  "loss": "{}".format(current_loss)})
            self.on_epoch_end(epoch=epoch, log={
                              "loss": "{}".format(current_loss)})

    def _fit_list(self, X: tf.Tensor):
        data_index = 0
        window_len = 2 * self.context_window + 1
        step = 0
        if not isinstance(X, tf.Tensor):
            raise TypeError("We were expecting a Tensor object!")
        # batch_size = self.batch_size
        data_len = len(X)
        # Note that we cannot fully digest all of the data in any one batch
        # if the window length is K and the natch_len is N, then the last
        # window that we get starts at position (N-K). Therefore, if we start
        # the next window at position (N-K)+1, we will get all windows.
        window_len = 1 + 2 * self.context_window
        shift_len = batch_size = window_len + 1
        # we need to make sure that we do not shift outside the boundaries of self.data too
        lastpos = data_len - 1  # index of the last word in data
        batch = 0
        for epoch in trange(self.epochs):
            data_index = 0
            endpos = data_index + batch_size
            while True:
                if endpos > lastpos:
                    break
                batch += 1
                currentTensor = X[data_index:endpos]
                if len(currentTensor) < window_len:
                    break  # We are at the end
                batch_x, batch_y = self.next_batch(currentTensor)
                current_loss = self.run_optimization(batch_x, batch_y)

                data_index += shift_len
                endpos = data_index + batch_size
                endpos = min(endpos,
                             lastpos)  # takes care of last part of data. Maybe we should just ignore though
                # Evaluation.
                step += 1
                self.run_optimization(batch_x, batch_y)
                self.on_batch_end(batch=batch, epoch=epoch, log={
                                  "loss": "{}".format(current_loss)})
            self.on_epoch_end(epoch=epoch, log={
                              "loss": "{}".format(current_loss)})
        self.on_training_end(log={})

    def fit(self, X: tf.Tensor, *args, samples_per_window: int = 2, context_window: int = 2, **kwargs):
        """Fit the Word2Vec continuous bag of words model.

        !TODO! Updated docstrings!

        Parameters
        ---------------------
        samples_per_window:
            samples_per_window: How many times to reuse an input to generate a label.

        """
        if samples_per_window > 2 * context_window:
            raise ValueError(
                'The value of self.samples_per_window must be <= twice the length of self.context_window')

        super().fit(X, *args, context_window=context_window, **kwargs)

        self. samples_per_window = samples_per_window

        if self.list_of_lists:
            # This is the case for a list of random walks
            # or for a list of text segments (e.g., a list of sentences or abstracts)
            self._fit_list_of_lists(X)
        else:
            self._fit_list(X)
