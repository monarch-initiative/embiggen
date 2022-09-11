"""Submodule providing wrapper for PyKEEN's NodePiece model."""
from typing import Union, Type, Dict, Any, List
from pykeen.training import TrainingLoop
from pykeen.models import NodePiece
from ensmallen import Graph
from embiggen.utils.abstract_models import EmbeddingResult
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKEEN
from pykeen.triples import CoreTriplesFactory


class NodePiecePyKEEN(EntityRelationEmbeddingModelPyKEEN):

    def __init__(
        self,
        embedding_size: int = 64,
        num_tokens: Union[int, List[int]] = 2,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        verbose: bool = False,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new PyKEEN NodePiece model.

        Details
        -------------------------
        This is a wrapper of the NodePiece implementation from the
        PyKEEN library. Please refer to the PyKEEN library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 64
            The dimension of the embedding to compute.
        num_tokens: Union[int, List[int]] = 2
            The number of relations to use to represent each entity, cf.
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
        ring_bell: bool = False,
            Whether to play a sound when embedding completes.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._num_tokens = num_tokens
        super().__init__(
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            training_loop=training_loop,
            verbose=verbose,
            random_state=random_state,
            ring_bell=ring_bell,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            num_tokens=self._num_tokens,
        )

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "NodePiece"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> NodePiece:
        """Build new NodePiece model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return NodePiece(
            triples_factory=triples_factory,
            num_tokens=self._num_tokens,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
    
    @classmethod
    def _create_inverse_triples(cls) -> bool:
        """Returns whether the class is expected to create inverse triples."""
        return True