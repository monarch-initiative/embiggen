"""Module providing adapter class making edge prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.edge_prediction.sklearn_like_edge_prediction_adapter import (
    SklearnLikeEdgePredictionAdapter,
)
from embiggen.utils.abstract_models import abstract_class


@abstract_class
class SklearnEdgePredictionAdapter(SklearnLikeEdgePredictionAdapter):
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_scale_free_distribution: bool = True,
        use_edge_metrics: bool = False,
        prediction_batch_size: int = 2**15,
        random_state: int = 42,
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        training_unbalance_rate: float = 1.0
            Unbalance rate for the training non-existing edges.
        training_sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges exclusively between nodes with different node types
            to generate the negative edges used during the training of the model.
            This can be useful when executing a bipartite edge prediction task.
        use_scale_free_distribution: bool = True
            Whether to sample the negative edges for the TRAINING of the model
            using a zipfian-like distribution that follows the degree distribution
            of the graph. This is generally useful, as these negative edges are less
            trivial to predict then edges sampled uniformely.
            We stringly advise AGAINST using uniform sampling.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        prediction_batch_size: int = 2**15
            Batch size to use for the predictions.
            Since usually rendering a whole dense graph edge embedding is not
            feaseable in main memory, we chunk it into more digestable smaller
            batches of edges.
        random_state: int
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
            edge_embedding_method=edge_embedding_method,
            training_unbalance_rate=training_unbalance_rate,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            use_scale_free_distribution=use_scale_free_distribution,
            use_edge_metrics=use_edge_metrics,
            prediction_batch_size=prediction_batch_size,
            random_state=random_state,
        )

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "scikit-learn"