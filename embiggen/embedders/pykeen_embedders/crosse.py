"""Submodule providing wrapper for PyKeen's CrossE model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import CrossE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class CrossEPyKeen(EntityRelationEmbeddingModelPyKeen):

    def __init__(
        self,
        embedding_size: int = 256,
        combination_dropout: float = 0.0,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        verbose: bool = False,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Create new PyKeen CrossE model.

        Details
        -------------------------
        This is a wrapper of the CrossE implementation from the
        PyKeen library. Please refer to the PyKeen library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 256
            The dimension of the embedding to compute.
        combination_dropout: float = 0.0
            The hidden dropout rate
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
        verbose: bool = False
            Whether to show loading bars.
        random_state: int = 42
            Random seed to use while training the model
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._combination_dropout = combination_dropout
        super().__init__(
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            training_loop=training_loop,
            verbose=verbose,
            random_state=random_state,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **dict(
                combination_dropout=self._combination_dropout,
            )
        )

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "CrossE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> CrossE:
        """Build new CrossE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return CrossE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            combination_dropout=self._combination_dropout,
            random_seed=self._random_state
        )
