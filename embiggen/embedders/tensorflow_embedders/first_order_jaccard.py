"""First order Jaccard TensorFlow model."""
from typing import Optional, Dict, Any

from ensmallen import Graph
from embiggen.sequences.tensorflow_sequences import EdgeJaccardSequence, AncestorsJaccardSequence
from embiggen.embedders.tensorflow_embedders.first_order_line import FirstOrderLINETensorFlow


class FirstOrderJaccard(FirstOrderLINETensorFlow):
    """First order Jaccard TensorFlow model."""

    def __init__(
        self,
        jaccard_type: str = "edge",
        root_node_name: Optional[str] = None,
        embedding_size: int = 100,
        negative_samples_rate: float = 0.5,
        epochs: int = 500,
        batch_size: int = 2**10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 5,
        use_mirrored_strategy: bool = False,
        optimizer: str = "nadam",
        enable_cache: bool = False,
        random_state: int = 42
    ):
        """Create new sequence Siamese model.

        Parameters
        -------------------------------------------
        jaccard_type: str = "edge"
            The Jaccard type to use. Can either be
            `edge`, for the traditional edge Jaccard,
            or alternatively the `ancestors` Jaccard,
            for which is mandatory to provide the root node.
        root_node_name: Optional[str] = None
            Root node to use when the ancestors mode for
            the Jaccard index is selected.
        embedding_size: int = 100
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        negative_samples_rate: float = 0.5
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        epochs: int = 10
            Number of epochs to train the model for.
        batch_size: int = 2**14
            Batch size to use during the training.
        early_stopping_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int = 1
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int = 1
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        optimizer: str = "nadam"
            The optimizer to be used during the training of the model.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        if root_node_name is None and jaccard_type == "ancestors":
            raise ValueError(
                "The provided jaccard type is `ancestors`, but "
                "the root node name was not provided."
            )
        if root_node_name is not None and jaccard_type == "edge":
            raise ValueError(
                "The provided jaccard type is `edge`, but "
                "the root node name was provided. It is unclear "
                "what to do with this parameter."
            )
        self._root_node_name = root_node_name
        self._jaccard_type = jaccard_type
        super().__init__(
            embedding_size=embedding_size,
            negative_samples_rate=negative_samples_rate,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            learning_rate_plateau_min_delta=learning_rate_plateau_min_delta,
            learning_rate_plateau_patience=learning_rate_plateau_patience,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            use_mirrored_strategy=use_mirrored_strategy,
            enable_cache=enable_cache,
            random_state=random_state
        )

    @staticmethod
    def model_name() -> str:
        """Returns name of the current model."""
        return "First Order Jaccard"

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                root_node_name=self._root_node_name,
                jaccard_type=self._jaccard_type,
            )
        }

    def _build_sequence(
        self,
        graph: Graph,
    ) -> EdgeJaccardSequence:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        if self._jaccard_type == "edge":
            return EdgeJaccardSequence(
                graph=graph,
                negative_samples_rate=self._negative_samples_rate,
                batch_size=self._batch_size,
                random_state=self._random_state
            )
        if self._jaccard_type == "ancestors":
            return AncestorsJaccardSequence(
                graph=graph,
                root_node_name=self._root_node_name,
                negative_samples_rate=self._negative_samples_rate,
                batch_size=self._batch_size,
                random_state=self._random_state
            )
        raise NotImplementedError(
            f"The provided Jaccard Index type {self._jaccard_type} "
            "is not currently supported."
        )
        