"""Abstract Torch/PyKeen Model wrapper for embedding models."""
from typing import Dict, List, Sequence, Union, Optional, Tuple, Any, Type

import numpy as np
import pandas as pd
from ensmallen import Graph
import inspect

from embiggen.utils.pytorch_utils import validate_torch_device
from embiggen.utils.abstract_models import AbstractEmbeddingModel, abstract_class, EmbeddingResult
from embiggen.utils.abstract_models import format_list
import torch
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop, TrainingLoop
from torch.optim import Optimizer


@abstract_class
class PyKeenEmbedder(AbstractEmbeddingModel):
    """Abstract Torch/PyKeen Model wrapper for embedding models."""

    SUPPORTED_TRAINING_LOOPS = {
        "Stochastic Local Closed World Assumption": SLCWATrainingLoop,
        "Local Closed World Assumption": LCWATrainingLoop,
    }

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 100,
        batch_size: int = 2**10,
        device: str = "auto",
        training_loop: Union[str, Type[TrainingLoop]
                             ] = "Stochastic Local Closed World Assumption",
        random_seed: int = 42,
        enable_cache: bool = False
    ):
        """Create new PyKeen Abstract Embedder model.
        
        Parameters
        -------------------------
        embedding_size: int = 100
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
        if isinstance(training_loop, str):
            if training_loop in PyKeenEmbedder.SUPPORTED_TRAINING_LOOPS:
                training_loop = PyKeenEmbedder.SUPPORTED_TRAINING_LOOPS[training_loop]
            else:
                raise ValueError(
                    f"The provided training loop name {training_loop} is not "
                    "a supported training loop name. "
                    f"The supported names are {format_list(PyKeenEmbedder.SUPPORTED_TRAINING_LOOPS)}."
                )

        if not inspect.isclass(training_loop):
            raise ValueError(
                "The provided training loop should be a class object.")

        if not issubclass(training_loop, TrainingLoop):
            raise ValueError(
                "The provided training loop class is not a subclass of `TrainingLoop` "
                f"and has type {type(training_loop)}."
            )

        self._training_loop = training_loop
        self._epochs = epochs
        self._batch_size = batch_size
        self._random_seed = random_seed
        self._device = validate_torch_device(device)

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=10,
            epochs=1
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                epochs=self._epochs,
                batch_size=self._batch_size,
                random_seed = self._random_seed
            )
        }

    @staticmethod
    def library_name() -> str:
        return "PyKeen"

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    def _build_model(self, triples_factory: CoreTriplesFactory) -> Type[Model]:
        """Build new model for embedding.

        Parameters
        ------------------
        triples_factory: CoreTriplesFactory
            The PyKeen triples factory to use to create the model.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__name__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_build_model`. Please do implement it."
        )

    def _get_steps_per_epoch(self, graph: Graph) -> Tuple[Any]:
        """Returns number of steps per epoch.

        Parameters
        ------------------
        graph: Graph
            The graph to compute the number of steps.
        """
        return None

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Type[Model],
        return_dataframe: bool
    ) -> EmbeddingResult:
        """Returns embedding from the model.

        Parameters
        ------------------
        graph: Graph
            The graph that was embedded.
        model: Type[Model]
            The Keras model used to embed the graph.
        return_dataframe: bool
            Whether to return a dataframe of a numpy array.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__name__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_extract_embeddings`. Please do implement it."
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Return node embedding"""

        torch_device = torch.device(self._device)

        triples_factory = CoreTriplesFactory(
            torch.IntTensor(graph.get_directed_edge_triples_ids().astype(np.int32)),
            num_entities=graph.get_nodes_number(),
            num_relations=graph.get_edge_types_number(),
            entity_ids=graph.get_node_ids(),
            relation_ids=graph.get_unique_edge_type_ids(),
            create_inverse_triples=False,
        )

        model = self._build_model(triples_factory)

        if not issubclass(model.__class__, Model):
            raise NotImplementedError(
                "The model created with the `_build_model` in the child "
                f"class {self.__class__.__name__} for the model {self.model_name()} "
                f"in the library {self.library_name()} did not return a "
                f"PyKeen model but an object of type {type(model)}."
            )

        # Move the model to gpu if we need to
        model.to(torch_device)

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
        )

        training_loop.train(
            triples_factory=triples_factory,
            num_epochs=self._epochs,
            batch_size=self._batch_size,
            use_tqdm=True,
            use_tqdm_batch=True,
            tqdm_kwargs=dict(
                disable=not verbose
            )
        )

        # Extract and return the embedding
        return self._extract_embeddings(
            graph,
            model,
            return_dataframe=return_dataframe
        )

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return False

    @staticmethod
    def is_topological() -> bool:
        return True

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return True

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return True

    @staticmethod
    def task_involves_edge_types() -> bool:
        """Returns whether the model task involves edge types."""
        return True