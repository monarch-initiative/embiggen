"""Module providing adapter class making node-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Optional
from ensmallen import Graph
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.node_label_prediction.sklearn_like_node_label_prediction_adapter import (
    SklearnLikeNodeLabelPredictionAdapter,
)
from embiggen.utils.abstract_models import abstract_class


@abstract_class
class SklearnNodeLabelPredictionAdapter(SklearnLikeNodeLabelPredictionAdapter):
    """Class wrapping Sklearn models for running node-label predictions."""

    def __init__(
        self, model_instance: Type[ClassifierMixin], random_state: Optional[int] = None
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into node-label prediction.
        random_state: Optional[int] = None
            The random state to use to reproduce the training.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        must_be_an_sklearn_classifier_model(model_instance)
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__
        super().__init__(
            model_instance=model_instance,
            random_state=random_state
        )

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "scikit-learn"
    
    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction."""
        return True
