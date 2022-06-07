"""Submodule providing wrapper for PyKeen's ComplEx model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import ComplEx
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class ComplExPyKeen(EntityRelationEmbeddingModelPyKeen):

    def __init__(
        self,
        embedding_size: int = 256,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        random_seed: int = 42,
        enable_cache: bool = False
    ):
        """Create new PyKeen ComplEx model.

        Details
        -------------------------
        This is a wrapper of the ComplEx implementation from the
        PyKeen library. Please refer to the PyKeen library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 256
            The dimension of the embedding to compute.
        epochs: int = 100
            The number of epochs to use to train the model for.
        batch_size: int = 2**10
            Size of the training batch.
        device: str = "auto"
            The devide to use to train the model.
            Can either be cpu or cuda.
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption"
            The training loop to use to train the model.
            Can either be:
            - Stochastic Local Closed World Assumption
            - Local Closed World Assumption
        random_seed: int = 42
            Random seed to use while training the model
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            training_loop=training_loop,
            random_seed=random_seed,
            enable_cache=enable_cache
        )

    @staticmethod
    def model_name() -> str:
        """Return name of the model."""
        return "ComplEx"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> ComplEx:
        """Build new ComplEx model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return ComplEx(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
        )
