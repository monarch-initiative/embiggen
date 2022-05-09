"""Module providing generic abstract model."""
from typing import Dict, Any


class AbstractModel:
    """Class defining properties of a generic abstract model."""

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        raise NotImplementedError((
            "The `parameters` method must be implemented "
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