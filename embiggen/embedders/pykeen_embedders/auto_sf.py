"""Submodule providing wrapper for PyKeen's AutoSF model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import AutoSF
from .entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class AutoSFPyKeen(EntityRelationEmbeddingModelPyKeen):

    def __init__(
        self,
        embedding_size: int = 256,
        num_components: int = 4,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption"
    ):
        """Create new PyKeen AutoSF model.

        Details
        -------------------------
        This is a wrapper of the AutoSF implementation from the
        PyKeen library. Please refer to the PyKeen library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 256
            The dimension of the embedding to compute.
        num_components: int = 4
            Number of components.
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
        self._num_components = num_components
        super().__init__(
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            training_loop=training_loop
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **EntityRelationEmbeddingModelPyKeen.smoke_test_parameters(),
            num_components=2
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                num_components=self._num_components
            )
        }

    @staticmethod
    def model_name() -> str:
        """Return name of the model."""
        return "AutoSF"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> AutoSF:
        """Build new AutoSF model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return AutoSF(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            num_components=self._num_components
        )
