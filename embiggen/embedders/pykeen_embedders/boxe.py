"""Submodule providing wrapper for PyKEEN's BoxE model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import BoxE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKEEN
from pykeen.triples import CoreTriplesFactory


class BoxEPyKEEN(EntityRelationEmbeddingModelPyKEEN):

    def __init__(
        self,
        embedding_size: int = 256,
        tanh_map: bool = True,
        p: int = 2,
        power_norm: bool = False,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        verbose: bool = False,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new PyKEEN BoxE model.

        Details
        -------------------------
        This is a wrapper of the BoxE implementation from the
        PyKEEN library. Please refer to the PyKEEN library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 256
            The dimension of the embedding to compute.
        tanh_map: bool = True
            Whether to use tanh mapping after BoxE computation (defaults to true). The hyperbolic tangent mapping
            restricts the embedding space to the range [-1, 1], and thus this map implicitly
            regularizes the space to prevent loss reduction by growing boxes arbitrarily large.
        p: int = 2
            order of norm in score computation
        param power_norm: bool = False
            whether to use the p-th power of the norm instead
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
        self._tanh_map=tanh_map
        self._p=p
        self._power_norm=power_norm
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
            **dict(
                tanh_map=self._tanh_map,
                p=self._p,
                power_norm=self._power_norm,
            )
        )

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "BoxE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> BoxE:
        """Build new BoxE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return BoxE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            tanh_map=self._tanh_map,
            p=self._p,
            power_norm=self._power_norm,
            random_seed=self._random_state
        )
