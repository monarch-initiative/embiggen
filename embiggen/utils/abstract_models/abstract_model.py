"""Module providing generic abstract model."""
from typing import Dict, Any
from dict_hash import Hashable, sha256


class AbstractModel(Hashable):
    """Class defining properties of a generic abstract model."""

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        raise NotImplementedError((
            "The `parameters` method must be implemented "
            "in the child classes of abstract model."
        ))

    def task_name(self) -> str:
        """Returns the task for which this model is being used."""
        raise NotImplementedError((
            "The `task_name` method must be implemented "
            "in the child classes of abstract model."
        ))
        
    def name(self) -> str:
        """Returns name of the model."""
        raise NotImplementedError((
            "The `name` method must be implemented "
            "in the child classes of abstract model."
        ))

    def clone(self) -> "Self":
        """Returns copy of the current model."""
        raise NotImplementedError((
            "The `clone` method must be implemented in "
            "the child classes of the abstract model."
        ))

    def consistent_hash(self)->str:
        """Returns consistent hash describing the model."""
        return sha256({
            **self.parameters(),
            "name": self.name(),
            "task_name": self.task_name(),
        })