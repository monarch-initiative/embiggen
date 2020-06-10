from typing import Tuple, Union, Dict


class Callback:
    """
    Superclass for embiggen callbacks
    """

    def __init__(self):
        """Superclass for embiggen callbacks.
        """
        super().__init__()

    def on_batch_start(self, epoch: int, batch: int, log: Dict[str, str] = None):
        """Callback for handling the batch's start event.

        Parameters
        ----------------------
        epoch: int,
            The current epoch number.
        batch: int,
            The current batch number.
        log: Dict,
            Additional informations relative to the current batch.
        """
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )

    def on_epoch_start(self, epoch: int, log: Dict[str, str] = None):
        """Callback for handling the epoch's start event.

        Parameters
        ----------------------
        epoch: int,
            The current epoch.
        log: Dict,
            Additional informations relative to the current epoch.
        """
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )

    def on_training_start(self):
        """Callback for handling the training's start event."""
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )

    def on_batch_end(self, epoch: int, batch: int, log: Dict[str, str] = None):
        """Callback for handling the batch's end event.

        Parameters
        ----------------------
        epoch: int,
            The current epoch number.
        batch: int,
            The current batch number.
        log: Dict,
            Additional informations relative to the current batch.
        """
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )

    def on_epoch_end(self, epoch: int, log: Dict[str, str] = None):
        """Callback for handling the epoch's end event.

        Parameters
        ----------------------
        epoch: int,
            The current epoch.
        log: Dict,
            Additional informations relative to the current epoch.
        """
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )

    def on_training_end(self):
        """Callback for handling the training's end event."""
        if self.__class__ == Callback:
            raise NotImplementedError(
                "The method must be implemented in the child classes of Callback."
            )
