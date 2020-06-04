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
