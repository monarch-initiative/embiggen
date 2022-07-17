"""Submodule providing wrapper for PyKeen's ConvE model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import ConvE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class ConvEPyKeen(EntityRelationEmbeddingModelPyKeen):

    def __init__(
        self,
        embedding_size: int = 256,
        input_channels: Optional[int] = None,
        output_channels: int = 32,
        embedding_height: Optional[int] = None,
        embedding_width: Optional[int] = None,
        kernel_height: int = 3,
        kernel_width: int = 3,
        input_dropout: float = 0.2,
        output_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        apply_batch_normalization: bool = True,
        epochs: int = 100,
        batch_size: int = 2**10,
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        verbose: bool = False,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Create new PyKeen ConvE model.

        Details
        -------------------------
        This is a wrapper of the ConvE implementation from the
        PyKeen library. Please refer to the PyKeen library documentation
        for details and posssible errors regarding this model.

        Parameters
        -------------------------
        embedding_size: int = 256
            The dimension of the embedding to compute.
        input_channels: Optional[int] = None
            the number of input channels for the convolution operation. Can be inferred from other parameters,
            cf. `_calculate_missing_shape_information`.
        output_channels: int = 32
            the number of input channels for the convolution operation
        embedding_height: Optional[int] = None
            the height of the "image" after reshaping the concatenated head and relation embedding. Can be inferred
            from other parameters, cf. `_calculate_missing_shape_information`.
        embedding_width: Optional[int] = None
            the width of the "image" after reshaping the concatenated head and relation embedding. Can be inferred
            from other parameters, cf. `_calculate_missing_shape_information`.
        kernel_height: int = 3
            the height of the convolution kernel. Defaults to `kernel_width`
        kernel_width: int = 3
            the width of the convolution kernel
        input_dropout: float = 0.2
            the dropout applied *before* the convolution
        output_dropout: float = 0.3
            the dropout applied after the linear projection
        feature_map_dropout: float = 0.2
            the dropout applied *after* the convolution
        apply_batch_normalization: bool = True
            whether to apply batch normalization
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
        self._input_channels=input_channels
        self._output_channels=output_channels
        self._embedding_height=embedding_height
        self._embedding_width=embedding_width
        self._kernel_height=kernel_height
        self._kernel_width=kernel_width
        self._input_dropout=input_dropout
        self._output_dropout=output_dropout
        self._feature_map_dropout=feature_map_dropout
        self._apply_batch_normalization=apply_batch_normalization
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
                input_channels=self._input_channels,
                output_channels=self._output_channels,
                embedding_height=self._embedding_height,
                embedding_width=self._embedding_width,
                kernel_height=self._kernel_height,
                kernel_width=self._kernel_width,
                input_dropout=self._input_dropout,
                output_dropout=self._output_dropout,
                feature_map_dropout=self._feature_map_dropout,
                apply_batch_normalization=self._apply_batch_normalization,
            )
        )

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "ConvE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> ConvE:
        """Build new ConvE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return ConvE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            input_channels=self._input_channels,
            output_channels=self._output_channels,
            embedding_height=self._embedding_height,
            embedding_width=self._embedding_width,
            kernel_height=self._kernel_height,
            kernel_width=self._kernel_width,
            input_dropout=self._input_dropout,
            output_dropout=self._output_dropout,
            feature_map_dropout=self._feature_map_dropout,
            apply_batch_normalization=self._apply_batch_normalization,
            random_seed=self._random_state
        )
