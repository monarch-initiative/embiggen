from .callback import Callback
from statistics import mean
from typing import Dict


class History(Callback):
    """
    This callback can be used to capture the history of the loss
    encountered during training. It takes the average loss over the
    last batch_count_for_avg batches and records it in a list which
    can be accessed after training.
    Parameters
        ---------------------
        batch_count_for_avg: int
            Determines how many loss values will be averaged per recorded
            data point
    """

    def __init__(self, batch_count_for_avg: int = 10):
        super().__init__()
        self.batch_count_for_avg = batch_count_for_avg
        self.loss_current = []
        self.loss_history = []

    def on_epoch_end(self, epoch: int, batch: int, log: Dict[str, str] = None):
        pass

    def on_batch_end(self, epoch: int, batch: int, log: Dict[str, str] = None):
        if 'loss' in log:
            loss = float(log['loss'])
            self.loss_current.append(loss)
        if len(self.loss_current) == self.batch_count_for_avg:
            meanloss = mean(self.loss_current)
            self.loss_history.append(meanloss)
            self.loss_current = []

    def on_training_end(self, epoch: int, batch: int, log: Dict[str, str] = None):
        for l in self.loss_history:
            print(l)

    def get_loss_history(self):
        return self.loss_history

    def get_batch_count_for_avg(self):
        """
        Get the number of batches averaged for one point of the history
        """
        return self.batch_count_for_avg
