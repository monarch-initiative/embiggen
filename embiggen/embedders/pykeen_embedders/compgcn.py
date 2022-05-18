"""Submodule providing wrapper for PyKeen's CompGCN model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import CompGCN
from .entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class CompGCNPyKeen(EntityRelationEmbeddingModelPyKeen):

    def __init__(
        self,
        embedding_size: int = 256,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption"
    ):
        """Create new PyKeen CompGCN model.

        Details
        -------------------------
        This is a wrapper of the CompGCN implementation from the
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
        """
        super().__init__(
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            training_loop=training_loop
        )

    @staticmethod
    def model_name() -> str:
        """Return name of the model."""
        return "CompGCN"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> CompGCN:
        """Build new CompGCN model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return CompGCN(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
        )
