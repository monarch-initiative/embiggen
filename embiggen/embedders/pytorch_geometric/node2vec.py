"""Abstract class for graph embedding models."""
from typing import Dict, Any

from embiggen.embedders.pytorch_geometric.pytorch_geometric_embedder import PyTorchGeometricEmbedder
from torch_geometric.nn import Node2Vec
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import DeviceObjType


class Node2VecPyTorchGeometric(PyTorchGeometricEmbedder):
    """Abstract class for sequence embedding models."""

    def __init__(
        self,

        embedding_size: int = 100,
        epochs: int = 30,
        number_of_negative_samples: int = 10,
        walk_length: int = 128,
        iterations: int = 10,
        window_size: int = 5,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        return_weight: float = 0.25,
        explore_weight: float = 4.0,
        random_state: int = 42,
        optimizer: str = "adam",
        verbose: bool = False,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new PyTorch Geometric Node2Vec model.

        Parameters
        -------------------------------
        
        """
        self._number_of_negative_samples = number_of_negative_samples
        self._embedding_size = embedding_size
        self._walk_length = walk_length
        self._window_size = window_size
        self._iterations = iterations
        self._return_weight = return_weight
        self._explore_weight = explore_weight

        self._loader = None

        super().__init__(
            random_state=random_state,
            embedding_size=embedding_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            verbose=verbose,
            ring_bell=ring_bell,
            enable_cache=enable_cache
        )

    def _build_model(
        self,
        edge_node_ids: Tensor,
        number_of_nodes: int
    ) -> Node2Vec:
        model = Node2Vec(
            edge_index=edge_node_ids,
            embedding_dim=self._embedding_size,
            walk_length=self._walk_length,
            context_size=2*self._window_size,
            walks_per_node=self._iterations,
            p=1.0/self._return_weight,
            q=1.0/self._explore_weight,
            num_negative_samples=self._number_of_negative_samples,
            num_nodes=number_of_nodes,
        )

        self._loader = model.loader(
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._number_of_workers
        )

        return model

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "Node2Vec"

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            walk_length=self._walk_length,
            window_size=self._window_size,
            iterations=self._iterations,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            number_of_negative_samples=self._number_of_negative_samples,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **super().smoke_test_parameters(),
            number_of_negative_samples=1,
            window_size=1,
            walk_length=2,
            iterations=1
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
        model.train()
        for pos_rw, neg_rw in self._loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
