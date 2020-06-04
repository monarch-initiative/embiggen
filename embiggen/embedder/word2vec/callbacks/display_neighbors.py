from .callback import Callback
from


class DisplayNeighbors(Callback):

    def __init__(
        self,
        X: np.ndarray,
        batch_steps: int = 1000,
        epochs_steps: int = 1
    ):
        """Create new instance of DisplayNeighbors.

        Parameters
        --------------------------
        steps: int = 1,
            Steps to wait between displaying the neighbors.

        """
        pass

    def display_words(self, x_test: np.array) -> None:
        """
        This is intended for to give feedback on the shell about the progress of training.
        It is not needed for the actual analysis.
        :param x_test:
        :return:
        """
        logging.info("Evaluation...")
        sim = calculate_cosine_similarity(get_embedding(x_test, self._embedding, self.device_type),
                                          self._embedding,
                                          self.device_type).numpy()
        # print(sim[0])
        for i in range(len(self.display_examples)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            disp_example = self.id2word[self.display_examples[i]]
            log_str = '"%s" nearest neighbors:' % disp_example
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, self.id2word[nearest[k]])
            # print(log_str)



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
            raise TypeError(
                'self.display requires a list of tuples with key:word, value:int (count)')

        if num > 16:
            logging.warning(
                'maximum of 16 display words allowed (you passed {num_words})'.format(num_words=num))
            num = 16

        # pick a random validation set of 'num' words to sample
        valid_window = 50
        valid_examples = np.array(random.sample(range(2, valid_window), num))

        # sample less common words - choose 'num' points randomly from the first 'valid_window' after element 1000
        self.display_examples = np.append(valid_examples, random.sample(
            range(1000, 1000 + valid_window), num), axis=0)

        return None