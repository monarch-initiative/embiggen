import argparse
from embiggen import Graph, GraphFactory, Embiggen
from embiggen.utils import write_embeddings, serialize, deserialize
from embiggen.neural_networks import MLP, FFNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from cache_decorator import Cache
from sanitize_ml_labels import sanitize_ml_labels
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import time
from tqdm.auto import tqdm
from typing import Dict, List
import compress_json


def get_current_dir() -> str:
    """Return the path to the folder of THIS file"""
    return os.path.dirname(os.path.abspath(__file__))


def cast_type(string: str):
    """Concretize the types from string to real types"""
    return {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str
    }[string.lower().strip()]


def parse_argument_json(path: str, parser: argparse.ArgumentParser):
    """Load the json and create the arguments"""
    args_settings = compress_json.load(path)

    for group, arguments in args_settings.items():
        # Create the group
        groups_settings = parser.add_argument_group(group)
        for _, description in arguments.items():
            # Concretize the type
            if "type" in description.keys():
                description["type"] = cast_type(description["type"])
            # Copy the settings to avoid mutation related problems
            settings = description.copy()
            # Split positional arguments from kwargs
            positional = []
            if "long" in settings.keys():
                positional.append(settings.pop("long"))
            if "short" in settings.keys():
                positional.append(settings.pop("short"))
            # Add the argument to the group
            groups_settings.add_argument(*positional, **settings)


def parse_args():
    """Parses arguments.

    """
    parser = argparse.ArgumentParser(description="Run link Prediction.")
    path = os.path.join(get_current_dir(), "cli_arguments.json")
    parse_argument_json(path, parser)
    return parser.parse_args()


def read_graphs(*paths: List[str], **kwargs: Dict) -> List[Graph]:
    """Return Graphs at given paths.

    These graphs are expected to be without a header.

    Parameters
    -----------------------
    *paths: List[str],
        List of the paths to be loaded.
        Notably, only the first one is fully preprocessed for random walks.
    **kwargs: Dict,

    """

    factory = GraphFactory()
    return [
        factory.read_csv(
            path,
            edge_has_header=False,
            start_nodes_column=0,
            end_nodes_column=1,
            weights_column=2,
            random_walk_preprocessing=i == 0,
            **kwargs
        )
        for i, path in tqdm(enumerate(paths), desc="Loading graphs")
    ]


def get_classifier_model(classifier: str, **kwargs: Dict):
    """Return choen classifier model.

    Parameters
    ------------------
    classifier:str,
        Chosen classifier model. Can either be:
            - LR for LogisticRegression
            - RF for RandomForestClassifier
            - MLP for Multi-Layer Perceptron
            - FFNN for Feed Forward Neural Network
    **kwargs:Dict,
        Keyword arguments to be passed to the constructor of the model.

    Raises
    ------------------
    ValueError,
        When given classifier model is not supported.

    Returns
    ------------------
    An instance of the selected model.
    """
    if classifier == "LR":
        return LogisticRegression(**kwargs)
    if classifier == "RF":
        return RandomForestClassifier(**kwargs)
    if classifier == "MLP":
        return MLP(**kwargs)
    if classifier == "FFNN":
        return FFNN(**kwargs)

    raise ValueError(
        "Given classifier model {} is not supported.".format(classifier)
    )


def performance_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return performance report for given predictions and ground truths.

    Parameters
    ------------------------
    y_true: np.ndarray,
        The ground truth labels.
    y_pred: np.ndarray,
        The labels predicted by the classifier.

    Returns
    ------------------------
    Dictionary with the performance metrics, including AUROC, AUPRC, F1 Score,
    and accuracy.
    """
    # TODO: add confusion matrix
    metrics = roc_auc_score, average_precision_score, f1_score
    report = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in metrics
    }
    report[sanitize_ml_labels(accuracy_score.__name__)] = accuracy_score(
        y_true, np.round(y_pred).astype(int)
    )
    return report


def main(args):
    """
    The input files are positive training, positive test, negative training and negative test edges. The code
    reads the files and create graphs in Graph format. Then, the positive training graph is embedded.
    Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Result of link prediction
    """

    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = read_graphs(
        args.pos_train,
        args.pos_valid,
        args.pos_test,
        args.neg_train,
        args.neg_valid,
        args.neg_test,
        return_weight=1/args.p,
        explore_weight=1/args.q
    )

    embedding = Embiggen()
    embedding.fit(
        pos_train,
        walks_number=args.walks_number,
        walks_length=args.walks_length,
        embedding_model=args.embedding_model,
        epochs=args.epochs,
        embedding_size=args.embedding_size,
        context_window=args.context_window,
        edges_embedding_method=args.edges_embedding_method
    )

    X_train, y_train = embedding.transform(pos_train, neg_train)
    X_test, y_test = embedding.transform(pos_test, neg_test)
    X_valid, y_valid = embedding.transform(pos_valid, neg_valid)

    classifier_model = get_classifier_model(
        args.classifier,
        **(
            dict(input_shape=X_train[:].shape[1])
            if args.classifier in ("MLP", "FFNN")
            else {}
        )
    )

    if args.classifier in ("MLP", "FFNN"):
        classifier_model.fit(X_train, y_train, X_test, y_test)
    else:
        classifier_model.fit(X_train, y_train)

    return dict(
        train=performance_report(y_train, classifier_model.predict(X_train)),
        test=performance_report(y_test, classifier_model.predict(X_test)),
        valid=performance_report(y_valid, classifier_model.predict(X_valid)),
    )


if __name__ == "__main__":
    args = parse_args()
    report = main(args)
    compress_json.dump(report, args.output_file, json_kwargs=dict(indent=4))
