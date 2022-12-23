"""Abstract Torch/PyTorch Geometric Model wrapper for embedding models."""
from typing import Dict, Union, Any, Type

import numpy as np
import pandas as pd
from ensmallen import Graph

from embiggen.utils.pytorch_utils import validate_torch_device
from embiggen.utils.abstract_models import AbstractEmbeddingModel, abstract_class, EmbeddingResult
import torch

from tqdm.auto import trange

from multiprocessing import cpu_count
from environments_utils import is_windows, is_macos
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import DeviceObjType


@abstract_class
class PyTorchGeometricEmbedder(AbstractEmbeddingModel):
    """Abstract PyTorch Geometric Model wrapper for embedding models."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 100,
        batch_size: int = 2**10,
        learning_rate: float = 0.01,
        number_of_workers: Union[int, str] = "auto",
        device: str = "auto",
        verbose: bool = False,
        random_state: int = 42,
        optimizer: str = "adam",
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new PyTorch Geometric Abstract Embedder model.
        
        Parameters
        -------------------------
        embedding_size: int = 100
            The dimension of the embedding to compute.
        epochs: int = 100
            The number of epochs to use to train the model for.
        batch_size: int = 2**10
            Size of the training batch.
        learning_rate: float = 0.01
            Learning rate of the model.
        device: str = "auto"
            The devide to use to train the model.
            Can either be cpu or cuda.
        verbose: bool = False
            Whether to show the loading bar.
        random_state: int = 42
            Random seed to use while training the model
        ring_bell: bool = False,
            Whether to play a sound when embedding completes.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._epochs = epochs
        self._verbose = verbose
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        if number_of_workers == "auto":
            number_of_workers = 0 if is_windows() or is_macos() else cpu_count()
        self._optimizer = optimizer
        self._number_of_workers = number_of_workers
        self._device = validate_torch_device(device)

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            ring_bell=ring_bell,
            random_state=random_state
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=10,
            epochs=1
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **dict(
                epochs=self._epochs,
                batch_size=self._batch_size,
            )
        )

    @classmethod
    def library_name(cls) -> str:
        return "PyTorch Geometric"

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    def _build_model(
        self,
        edge_node_ids: Tensor,
        number_of_nodes: int
    ) -> Type[Module]:
        """Build new model for embedding.

        Parameters
        ------------------
        edge_node_ids: Tensor
            Tuples with the source and destination node ids
        number_of_nodes: int
            Number of nodes in the graph.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__class__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_build_model`. Please do implement it."
        )

    def _train_model_step(
        self,
        model: Module,
        optimizer: Optimizer,
        device: DeviceObjType
    ):
        """Train model for a single step.

        Parameters
        ------------------
        model: Module
            The model to be trained for this step.
        optimizer: Optimizer
            The optimizer to be trained for this step.
        device: DeviceObjType
            The device to be used.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__class__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_train_model_step`. Please do implement it."
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Return node embedding"""

        torch_device = torch.device(self._device)

        edge_node_ids = torch.LongTensor(np.int64(graph.get_directed_edge_node_ids().T))

        model = self._build_model(
            edge_node_ids=edge_node_ids,
            number_of_nodes=graph.get_number_of_nodes()
        )

         # Move the model to gpu if we need to
        model.to(torch_device)

        if not issubclass(model.__class__, Module):
            raise NotImplementedError(
                "The model created with the `_build_model` in the child "
                f"class {self.__class__.__name__} for the model {self.model_name()} "
                f"in the library {self.library_name()} did not return a "
                f"PyTorch Geometric model but an object of type {type(model)}."
            )
        
        optimizer = torch.optim.Adam(
            list(model.parameters()),
            lr=self._learning_rate
        )

        for _ in trange(
            self._epochs,
            dynamic_ncols=True,
            desc="Epochs",
            disable=not self._verbose,
            leave=False
        ):
            self._train_model_step(
                model=model,
                optimizer=optimizer,
                device=torch_device
            )

        # Extract and return the embedding
        node_embedding = model(torch.arange(
            graph.get_number_of_nodes(),
            device=torch_device
        )).cpu().detach().numpy()

        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can use node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can use edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True