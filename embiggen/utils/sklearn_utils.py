"""Submodule with utils for interface with Sklearn models."""
import functools
from typing import Dict, Type
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from userinput.utils import closest
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
    "MLPClassifier": get_default_mlp_classifier
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


def evaluate_sklearn_classifier(
    classifier: Type[ClassifierMixin],
    X: np.ndarray,
    y: np.ndarray,
    multiclass_or_multilabel: bool,
    **kwargs: Dict
) -> Dict[str, float]:
    """Return dict with evaluation of the model on classification task.

    Parameters
    --------------
    classifier: Type[ClassifierMixin]
        The classifier to evaluate.
    X: np.ndarray
        The input data.
    y: np.ndarray
        The output labels
    multiclass_or_multilabel: bool
        Whether the provided labels are multiclass or multilabel,
        that is the model needs to compute more output
        classes for a given input.
    **kwargs: Dict
        Aguments to forward to the model predict proba method.
    """

    must_be_an_sklearn_classifier_model(classifier)

    predictions = classifier.predict_proba(X, **kwargs)

    # If this is a binary prediction, we need to handle
    # this as a corner case.
    if not multiclass_or_multilabel:
        predictions = predictions[:, 1]

    integer_predictions = classifier.predict(X, **kwargs)

    # Depending on whether this task is binary or multiclall/multilabel
    # we need to set the average parameter for the metrics.
    if multiclass_or_multilabel:
        average_methods = average_methods_probs = ("macro", "weighted")
    else:
        average_methods = ("binary",)
        average_methods_probs = ("micro", "macro", "weighted")

    @functools.wraps(roc_auc_score)
    def wrapper_roc_auc_score(*args, **kwargs):
        return roc_auc_score(*args, **kwargs, multi_class="ovr")

    return {
        **{
            sanitize_ml_labels(integer_metric.__name__): integer_metric(y, integer_predictions)
            for integer_metric in (
                accuracy_score,
                balanced_accuracy_score,
            )
        },
        **{
            "{}{}".format(
                sanitize_ml_labels(averaged_integer_metric.__name__),
                " ".format(
                    average_method) if average_method != "binary" else ""
            ): averaged_integer_metric(y, integer_predictions, average=average_method)
            for averaged_integer_metric in (
                f1_score,
                precision_score,
                recall_score,
            )
            for average_method in average_methods
        },
        **{
            "{} {}".format(
                sanitize_ml_labels(probabilistic_metric.__name__),
                average_method
            ): probabilistic_metric(
                y,
                predictions,
                average=average_method
            )
            for probabilistic_metric in (
                # AUPRC in sklearn is only supported for binary labels
                *((average_precision_score,)
                  if not multiclass_or_multilabel else ()),
                wrapper_roc_auc_score
            )
            for average_method in average_methods_probs
        },
    }
