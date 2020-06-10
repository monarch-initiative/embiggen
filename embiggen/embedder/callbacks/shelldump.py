from typing import Tuple, Union, Dict

from .callback import Callback


class Shelldump(Callback):
    """
    This callback can be used to display messages during training.
    It is primarily useful for developing and debugging but is not
    necessary for training.
    Parameters
        ---------------------
        batch_display_interval: int
            Determines how often we will display a message during training 
            If the value is None, we display nothing
            If the value is n, we print a display to shell every n batches
    """
    def __init__(self,batch_display_interval: int = 1000):
        super().__init__()
        self.batch_display_interval = batch_display_interval

    def on_epoch_end(self,logs:Dict[str,str]=None):
        super().on_epoch_end(logs)
        for k,v in logs:
            print("{}:{}".format(k,v))

    def on_batch_end(self,logs:Dict[str,str]=None):
        pass

    def on_training_end(self,logs:Dict[str,str]=None):
        pass