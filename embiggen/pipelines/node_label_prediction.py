"""Submodule providing pipelines for node-label evaluation and prediction."""
from typing import Union, Callable, List, Optional, Dict, Generator
import pandas as pd
import numpy as np
from ensmallen import Graph
from time import time
from tqdm.auto import trange, tqdm
from ensmallen.datasets import get_dataset
from sklearn.base import ClassifierMixin
import copy
from ..node_label_prediction import SklearnModelNodeLabelPredictionAdapter
from ..utils import is_sklearn_classifier_model, get_sklearn_default_classifier, is_default_sklearn_classifier
from .compute_node_embedding import compute_node_embedding


def _node_embedding_depends_on_node_types(node_embedding_method_name: str) -> bool:
    """Returns whether a given node embedding method name depends on the graph node types.

    The implications of such a dependency are that we can either compute the node embedding
    once for each provided graph when the start the pipeline, or we need to compute it
    multiple times, once per each of the holdouts.

    Parameters
    -------------------------
    node_embedding_method_name: str
        The method to check for.
    """
    # TODO: actually implement this method when other methods such as Siamese will start getting used.
    return False


def _compute_node_features(
    graph: Graph,
    compute_node_type_dependent_features: bool,
    node_features: Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]],
    embedding_kwargs: Optional[Dict] = None,
    embedding_fit_kwargs: Optional[Dict] = None,
) -> List[Union[str, pd.DataFrame, np.ndarray]]:
    """Computes and returns all possible node features.

    Parameters
    ---------------------
    graph: Graph
        The graph to compute the features for.
    compute_node_type_dependent_features: bool
        Whether we can currently compute the node features that are
        dependent on the node types.
    node_features: Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        The node features to compute.
    embedding_kwargs: Optional[Dict] = None
        Kwargs to forward to the copmputation of the embedding.
    embedding_fit_kwargs: Optional[Dict] = None
        Kwargs to forward to the fit portion of the embedding.
    """
    if embedding_kwargs is None:
        embedding_kwargs = {}

    if not isinstance(node_features, list):
        node_features = [node_features]
    return [
        (
            compute_node_embedding(
                graph=graph,
                node_embedding_method_name=node_feature,
                fit_kwargs=embedding_fit_kwargs,
                verbose=False,
                **embedding_kwargs
            )[0]
            if not compute_node_type_dependent_features or _node_embedding_depends_on_node_types(node_feature)
            else node_feature
        ) if isinstance(node_feature, str) else node_feature
        for node_feature in node_features
    ]


def _validate_and_align_node_feature(
    graph: Graph,
    node_features: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Verifies that the node features provided are compatible with this graph

    Parameters
    ---------------------
    graph: Graph
        The graph to check for.
    node_features: Union[pd.DataFrame, np.ndarray]
        The features to validate and align.
    """
    if not isinstance(node_features, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            (
                "The provided node features are of type {node_features_type}, "
                "while we only currently support numpy arrays and pandas DataFrames. "
                "What behaviour were you expecting with this feature? "
                "Please do open an issue on Embiggen and let us know!"
            ).format(
                node_features_type=type(node_features)
            )
        )

    if graph.get_nodes_number() != node_features.shape[0]:
        raise ValueError(
            (
                "The provided node features have {rows_number} rows "
                "but the provided graph{graph_name} has {nodes_number} nodes. "
                "Maybe these features refer to another "
                "version of the graph or another graph "
                "entirely?"
            ).format(
                rows_number=node_features.shape[0],
                graph_name="" if graph.get_name().lower(
                ) == "graph" else " {}".format(graph.get_name()),
                nodes_number=graph.get_nodes_number()
            )
        )

    # If it is a dataframe we align it
    if isinstance(node_features, pd.DataFrame):
        return node_features.loc[graph.get_node_names()].to_numpy()

    # And if it is a numpy array we must believe that the user knows what
    # they are doing, as we cannot ensure alignment.
    return node_features


def _validate_and_align_node_features(
    graph: Graph,
    node_features: List[Union[pd.DataFrame, np.ndarray]],
) -> List[np.ndarray]:
    """Verifies that the node features provided are compatible with this graph

    Parameters
    ---------------------
    graph: Graph
        The graph to check for.
    node_features: List[Union[pd.DataFrame, np.ndarray]]
        The features to validate and align.
    """
    return [
        _validate_and_align_node_feature(graph, nfs)
        for nfs in node_features
    ]


def _get_node_label_classifier_model(model, **kwargs: Dict):
    """Returns a new node label classifier model that may be trained.

    Parameters
    ------------------------
    model
        The model describing what the new model should look like.
        This object can be:
        - A string with the name of the desired model.
        - An instantiated Sklearn model that will be cloned and wrapped.
    **kwargs: Dict
        The kwargs to provide to the model constructor.
    """
    # If the provided model is a string
    if isinstance(model, str):
        if is_default_sklearn_classifier(model):
            return SklearnModelNodeLabelPredictionAdapter(
                get_sklearn_default_classifier(
                    model,
                    **kwargs
                )
            )
        raise ValueError(
            "The provided model name {} is not available.".format(
                model
            )
        )

    # If the provide model is already an sklearn model,
    # we proceed to wrap it and we avoid to training the original
    # models multiple times by making a deep copy of it.
    if is_sklearn_classifier_model(model):
        if kwargs:
            raise ValueError(
                "Kwargs for instantating a model were provided but "
                "the provided model is already instantiated. It is not "
                "clear what to do with these arguments."
            )
        return SklearnModelNodeLabelPredictionAdapter(
            copy.deepcopy(model)
        )

    raise ValueError(
        "It is not clear what to do with the provided model object of type {}.".format(
            type(model)
        )
    )

# TODO! Add cache!


def _execute_node_label_prediction_holdout(
    graph: Graph,
    graph_name: str,
    classifier_instance,
    number_of_holdouts: int,
    holdout_number: int,
    train_size: float,
    holdout_type: str,
    use_stratification: bool,
    random_state: int,
    graph_node_features: List[Union[str, pd.DataFrame, np.ndarray]],
    embedding_kwargs: Dict,
    embedding_fit_kwargs: Dict,
    classifier_kwargs: Dict,
    classifier_fit_kwargs: Dict
) -> List[Dict]:
    """Execute an holdout."""
    # Split the graph using the requested holdout type.
    if holdout_type == "Monte Carlo":
        train_graph, test_graph = graph.get_node_label_holdout_graphs(
            train_size=train_size,
            use_stratification=use_stratification,
            random_state=random_state*holdout_number
        )
    elif holdout_type == "KFold":
        train_graph, test_graph = graph.get_node_label_kfold(
            k=number_of_holdouts,
            k_index=holdout_number,
            use_stratification=use_stratification,
            random_state=random_state*holdout_number
        )
    else:
        raise ValueError(
            "The provided holdout type is not supported. "
            "We currently only support Monte Carlo and Kfold."
        )

    # Compute the remaining node features that we could not
    # compute before because of their dependence on the node types,
    # which would have caused a bias in the evaluation.
    # Once the features are computed, we also run a validation
    # on them to make sure that indeed they are compatible with
    # the provided graph.
    holdout_node_features = _validate_and_align_node_features(
        train_graph,
        _compute_node_features(
            train_graph,
            compute_node_type_dependent_features=True,
            node_features=graph_node_features,
            embedding_kwargs=embedding_kwargs,
            embedding_fit_kwargs=embedding_fit_kwargs
        )
    )

    start_training = time()
    classifier_instance.fit(
        graph=train_graph,
        node_features=holdout_node_features,
        behaviour_for_unknown_node_labels="drop",
        aligned_node_mapping=True,
        random_state=random_state,
    )
    training_time = time() - start_training

    both_performance = []

    for evaluation_type, graph_to_evaluate in (
        ("train", train_graph),
        ("test", test_graph),
    ):
        start_evaluation = time()
        performance = classifier_instance.evaluate(
            graph=graph_to_evaluate,
            node_features=holdout_node_features,
            behaviour_for_unknown_node_labels="drop",
            aligned_node_mapping=True,
            random_state=random_state,
        )
        evaluation_time = time() - start_evaluation

        # Add more metadata to the evaluation report.
        performance["evaluation_type"] = evaluation_type
        performance["size"] = train_size
        performance["graph_name"] = graph_name
        performance["holdout_type"] = holdout_type
        performance["required_training_time"] = training_time
        performance["required_evaluation_time"] = evaluation_time

        both_performance.append(performance)

    return both_performance


def _get_graphs_iterator(
    graphs: Union[List[Graph], List[str], Graph, str],
    repositories: Union[str, List[str]],
    versions: Union[str, List[str]],
    graph_normalization_callback: Callable[[Graph], Graph],
) -> List[Graph]:
    """Returns iterator on given graph objects.

    Parameters
    ------------------
    graphs: Union[Graph, str, List[str], List[Graph]]
        The graph to run the embedding and node-label prediction on.
        If a string was provided, we will retrieve the graphs from Ensmallen's repositories.
        If a list was provided, we will iterate on all graphs.
    repositories: Union[str, List[str]]
        Repositorie(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    versions: Union[str, List[str]]
        Version(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    graph_normalization_callback: Callable[[Graph], Graph]
        Graph normalization procedure to call on graphs that have been loaded from
        the Ensmallen automatic retrieval.
    """
    if not isinstance(graphs, list):
        graphs = [graphs]

    number_of_graphs = len(graphs)

    if versions is not None:
        if any(not isinstance(graph, str) for graph in graphs):
            raise ValueError(
                "Graph versions were provided, but the graphs are not ",
                "graph names from Ensmallen's automatic retrieval."
            )
        if isinstance(versions, str):
            versions = [versions] * number_of_graphs
    else:
        versions = [None] * number_of_graphs

    if repositories is not None:
        if any(not isinstance(graph, str) for graph in graphs):
            raise ValueError(
                "Graph repositories were provided, but the graphs are not ",
                "graph names from Ensmallen's automatic retrieval."
            )
        if isinstance(repositories, str):
            repositories = [repositories]*len(graphs)
    else:
        repositories = [None] * number_of_graphs

    for graph, repository, version in tqdm(
        zip(graphs, repositories, versions),
        desc="Running node-label prediction evaluations",
        disable=number_of_graphs == 1,
        total=number_of_graphs,
        dynamic_ncols=True,
        leave=False
    ):
        if isinstance(graph, str):
            graph = get_dataset(
                graph,
                repository=repository,
                version=version
            )()
            if graph_normalization_callback is not None:
                yield graph_normalization_callback(graph)
        yield graph


# TODO! Add caching decorator!
def _run_node_label_prediction_on_graph(
    graph: Graph,
    models: Union[str, Callable, ClassifierMixin, List[Union[str, Callable, ClassifierMixin]]],
    node_features: Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]],
    number_of_holdouts: int,
    train_size: float,
    holdout_type: str,
    use_stratification: bool,
    random_state: int,
    embedding_kwargs: Optional[Dict],
    embedding_fit_kwargs: Optional[Dict],
    classifier_kwargs: Optional[Dict],
    classifier_fit_kwargs: Optional[Dict],
) -> List[Dict]:
    """Run the experiment on the provided graph."""
    # We compute all the node features we can
    # that are not dependent on the node types and therefore
    # do not cause a bias while evaluating them.
    graph_node_features = _compute_node_features(
        graph,
        compute_node_type_dependent_features=False,
        node_features=node_features,
        embedding_kwargs=embedding_kwargs,
        embedding_fit_kwargs=embedding_fit_kwargs
    )

    if classifier_kwargs is None:
        classifier_kwargs = {}

    if classifier_fit_kwargs is None:
        classifier_fit_kwargs = {}

    if not isinstance(models, list):
        models = [models]

    # And for each graph we compute a k-fold holdout
    return [
        performance
        for holdout_number in trange(
            number_of_holdouts,
            desc=f"Node-label prediction holdouts on {graph.get_name()}",
            dynamic_ncols=True,
            leave=False
        )
        # Iterate over the provided models
        for model in models
        # We compute the training and test performance on this holdout.
        for performance in _execute_node_label_prediction_holdout(
            graph,
            graph_name=graph.get_name(),
            classifier_instance=_get_node_label_classifier_model(
                model,
                **classifier_kwargs
            ),
            number_of_holdouts=number_of_holdouts,
            holdout_number=holdout_number,
            train_size=train_size,
            holdout_type=holdout_type,
            use_stratification=use_stratification,
            random_state=random_state,
            graph_node_features=graph_node_features,
            embedding_kwargs=embedding_kwargs,
            embedding_fit_kwargs=embedding_fit_kwargs,
            classifier_kwargs=classifier_kwargs,
            classifier_fit_kwargs=classifier_fit_kwargs
        )
    ]

# TODO! Add cache decorator!


def node_label_prediction(
    node_features: Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]],
    graphs: Union[Graph, str, List[str], List[Graph]],
    models: Union[str, Callable, ClassifierMixin, List[Union[str, Callable, ClassifierMixin]]] = "DecisionTreeClassifier",
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    number_of_holdouts: int = 10,
    train_size: float = 0.8,
    holdout_type: str = "Monte Carlo",
    use_stratification: bool = True,
    random_state: int = 42,
    embedding_kwargs: Optional[Dict] = None,
    embedding_fit_kwargs: Optional[Dict] = None,
    classifier_kwargs: Optional[Dict] = None,
    classifier_fit_kwargs: Optional[Dict] = None,
    graph_normalization_callback: Callable[[Graph], Graph] = None,
) -> pd.DataFrame:
    """Return the evaluation of an embedding for edge prediction on the given model.

    Parameters
    ----------------------
    node_features: Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        The node features to be used to train the classifier.
        If a string is provided, the curresponding embedding will be computed.
        The features will be provided to the model always as a list.
    graphs: Union[Graph, str, List[str], List[Graph]]
        The graph to run the embedding and node-label prediction on.
        If a string was provided, we will retrieve the graphs from Ensmallen's repositories.
        If a list was provided, we will iterate on all graphs.
    model: Union[str, Callable, ClassifierMixin]
        Either the name of the model or a method returning a model.
    repositories: Optional[Union[str, List[str]]] = None
        Repositorie(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    versions: Optional[Union[str, List[str]]] = None
        Version(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    number_of_holdouts: int = 10
        The number of the holdouts to run.
    train_size: float = 0.8,
        Split size of the training graph.
        This value will be ignored if the holdouts is
        specified to be executed as a kfold.
    holdout_type: str = "Monte Carlo"
        Whether to use a Monte Carlo holdout or a Kfold
        holdout. The possible values are:
        - "Monte Carlo": holdout with repeated sampling, the different holdouts share the same complete set of samples.
        - "K fold": splits the dataset into k folds.
    use_stratification: bool = True
        Whether to use stratification.
        It is generally a VERY POOR CHOICE when running
        experiments to not use stratification as it may
        lead to biases dependending on your dataset labels
        distribution.
    random_state: int = 42
        The seed to be used to reproduce the holdout.
    embedding_fit_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding fit method
    embedding_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding method
    graph_normalization_callback: Callable[[Graph], Graph] = None
        Graph normalization procedure to call on graphs that have been loaded from
        the Ensmallen automatic retrieval.
    """
    return pd.DataFrame([
        holdout
        # We iterate over the provided graphs
        for graph in _get_graphs_iterator(
            graphs=graphs,
            repositories=repositories,
            versions=versions,
            graph_normalization_callback=graph_normalization_callback
        )
        for holdout in _run_node_label_prediction_on_graph(
            graph=graph,
            models=models,
            node_features=node_features,
            number_of_holdouts=number_of_holdouts,
            train_size=train_size,
            holdout_type=holdout_type,
            use_stratification=use_stratification,
            random_state=random_state,
            embedding_kwargs=embedding_kwargs,
            embedding_fit_kwargs=embedding_fit_kwargs,
            classifier_kwargs=classifier_kwargs,
            classifier_fit_kwargs=classifier_fit_kwargs,
        )
    ])
