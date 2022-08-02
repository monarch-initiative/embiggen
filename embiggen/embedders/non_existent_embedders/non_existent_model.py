"""A model strictly for validating the meta programming."""
from embiggen.utils.abstract_models import AbstractEmbeddingModel


class NonExistentModel(AbstractEmbeddingModel):
    """A model strictly for validating the meta programming."""

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Non Existent Model"

import non_existent_module
    