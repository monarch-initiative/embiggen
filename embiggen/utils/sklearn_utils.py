"""Submodule with utils for interface with Sklearn models."""
import functools
from typing import Dict, Type
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from userinput.utils import closest
import numpy as np


def get_default_logistic_regression_classifier(**kwargs: Dict) -> LogisticRegression:
    """Returns a default logistic regression classifier.

    On the parameters used
    ----------------------------
    The parameters used for this model are reasonable
    general use parameters, but if you had any better
    general parameters please do feel free to drop an
    issue on the Embiggen repository with your proposal.
    Your insight is absolutely welcomed.
    """
    return LogisticRegression(**{
        **dict(
            n_jobs=-1,
            random_state=42
        ),
        **kwargs
    })


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
            n_neighbors=10,
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
    "DecisionTreeClassifier": get_default_decision_tree_classifier,
    "ExtraTreesClassifier": get_default_extra_tree_classifier,
    "RandomForestClassifier": get_default_random_forest_classifier,
    "KNeighborsClassifier": get_default_k_neighbours_classifier,
    "MLPClassifier": get_default_mlp_classifier,
    "LogisticRegression": get_default_logistic_regression_classifier
}


def is_default_sklearn_classifier(model_name: str) -> bool:
    """Returns whether we have default models for the provided model name.

    Parameters
    ------------------------
    model_name: str
        Name of the default model to check for.
    """
    return model_name.lower() in {
        key.lower()
        for key in available_default_sklearn_classifiers
    }


def must_be_default_sklearn_classifier(model_name: str) -> bool:
    """Raises exception if we do not have default models for the provided model name.

    Parameters
    ------------------------
    model_name: str
        Name of the default model to check for.

    Raises
    ------------------------
    ValueError
        If the provided model is not currently available as a default.
    """
    if not is_default_sklearn_classifier(model_name):
        raise ValueError(
            (
                "The provided model name `{model_name}` is not a supported sklearn "
                "classifier among the default available ones. Possibly you meant {candidate}?"
            ).format(
                model_name=model_name,
                candidate=closest(
                    model_name, available_default_sklearn_classifiers.keys())
            )
        )


def get_sklearn_default_classifier(model_name: str, **kwargs: Dict):
    """Returns instance of default Sklearn classifier.

    Parameters
    ------------------------
    model_name: str
        Name of the default model to load.
    **kwargs: Dict
        Arguments to forward to the sklearn model construction.
    """
    must_be_default_sklearn_classifier(model_name)
    return {
        key.lower(): callback
        for key, callback in available_default_sklearn_classifiers.items()
    }[model_name.lower()](**kwargs)


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
