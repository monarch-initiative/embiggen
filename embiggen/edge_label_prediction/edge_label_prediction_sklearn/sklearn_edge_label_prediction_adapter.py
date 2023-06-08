"""Module providing adapter class making edge-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)


class SklearnEdgeLabelPredictionAdapter(SklearnLikeEdgeLabelPredictionAdapter):
    """Class wrapping Sklearn models for running edge-label prediction."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge-label prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        random_state: int = 42
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
            random_state=random_state,
            edge_embedding_method=edge_embedding_method,
            use_edge_metrics=use_edge_metrics,
        )

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "scikit-learn"