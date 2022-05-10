"""Submodule with utils for interface with Sklearn models."""
from sklearn.base import ClassifierMixin


def is_sklearn_classifier_model(candidate_model) -> bool:
    """Returns whether a given object is a Sklearn classifier model."""
    return issubclass(candidate_model.__class__, ClassifierMixin) and all(
        hasattr(candidate_model, method_name)
        for method_name in (
            "predict_proba",
            "predict",
            "fit",
        )
    )


def must_be_an_sklearn_classifier_model(candidate_model):
    """Raises an exception if the provided object is not an sklearn classifier model."""
    if not is_sklearn_classifier_model(candidate_model):
        raise ValueError(
            (
                "The provided object of type {} is not a valid sklearn classifier model "
                "that can be adapted for this class as it is not a subclass of `ClassifierMixin`."
            ).format(type(candidate_model))
        )
