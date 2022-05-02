"""Submodule with utils for interface with Sklearn models."""
from typing import Dict, Type
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def get_default_decision_tree_classifier(**kwargs: Dict) -> DecisionTreeClassifier:
    """Returns a default decision tree classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return DecisionTreeClassifier(**{
        **dict(
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),
        **kwargs
    })


def get_default_extra_tree_classifier(**kwargs: Dict) -> ExtraTreesClassifier:
    """Returns a default extra tree classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return ExtraTreesClassifier(**{
        **dict(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            bootstrap=True,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced"
        ),
        **kwargs
    })


def get_default_random_forest_classifier(**kwargs: Dict) -> RandomForestClassifier:
    """Returns a default random forest classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return RandomForestClassifier(**{
        **dict(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            bootstrap=True,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced"
        ),
        **kwargs
    })


def get_default_k_neighbours_classifier(**kwargs: Dict) -> KNeighborsClassifier:
    """Returns a default K-neighbours classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return KNeighborsClassifier(**{
        **dict(
            n_estimators=10,
            weights="distance",
            n_jobs=-1,
        ),
        **kwargs
    })


def get_default_mlp_classifier(**kwargs: Dict) -> MLPClassifier:
    """Returns a default MLP classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return MLPClassifier(**{
        **dict(
            activation="relu",
            hidden_layer_sizes=100,
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
        ),
        **kwargs
    })


available_default_sklearn_classifiers = {
    "DecisionTreeClassifier".lower(): get_default_decision_tree_classifier,
    "ExtraTreesClassifier".lower(): get_default_extra_tree_classifier,
    "RandomForestClassifier".lower(): get_default_random_forest_classifier,
    "KNeighborsClassifier".lower(): get_default_k_neighbours_classifier,
    "MLPClassifier".lower(): get_default_mlp_classifier
}


def is_default_sklearn_classifier(model_name: str) -> bool:
    return model_name.lower() in available_default_sklearn_classifiers


def get_sklearn_default_classifier(model_name: str, **kwargs):
    return available_default_sklearn_classifiers[model_name.lower()](**kwargs)


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


def evaluate_sklearn_classifier(
    classifier: Type[ClassifierMixin],
    X: np.ndarray,
    y: np.ndarray,
    **kwargs: Dict
) -> Dict[str, float]:
    """Return dict with evaluation of the model on classification task."""

    must_be_an_sklearn_classifier_model(classifier)

    predictions = classifier.predict_proba(X, **kwargs)

    # If this is a binary prediction, we need to handle
    # this as a corner case.
    if len(y.shape) == 1:
        predictions = predictions[:, 1]

    integer_predictions = classifier.predict(X, **kwargs)

    return {
        **{
            sanitize_ml_labels(integer_metric.__name__): integer_metric(y, integer_predictions)
            for integer_metric in (
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )
        },
        **{
            sanitize_ml_labels(probabilistic_metric.__name__): probabilistic_metric(y, predictions)
            for probabilistic_metric in (
                average_precision_score,
                roc_auc_score
            )
        },
    }
