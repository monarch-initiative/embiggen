from typing import Tuple, Union, Dict

class Callback:
    """
    Superclass for embiggen callbacks
    """
    def __init__(self):
        """Superclass for embiggen callbacks.
        """
        super().__init__()
        self.batch_number = 0 # number of completed batches
        self.epoch_number = 0 # number of completed epochs
        

    def on_epoch_end(self):
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            ) 

    def on_batch_end(self):
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            ) 

    def on_training_end(self):
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            ) 