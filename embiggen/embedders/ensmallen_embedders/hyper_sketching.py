"""Module providing HyperSketching implementation."""
from typing import Dict, Any, Optional, Tuple, List
from ensmallen import Graph
import pandas as pd
import numpy as np
import compress_json
import json
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult
from embiggen.utils import AbstractEdgeFeature


class HyperSketching(EnsmallenEmbedder, AbstractEdgeFeature):
    """Class implementing the HyperSketching edge embedding method."""

    def __init__(
        self,
        number_of_hops: int = 3,
        precision: int = 6,
        bits: int = 4,
        include_node_types: bool = False,
        include_edge_types: bool = False,
        include_edge_ids: bool = False,
        include_node_ids: bool = True,
        include_selfloops: bool = True,
        include_typed_graphlets: bool = False,
        normalize_by_symmetric_laplacian: bool = True,
        concatenate_features: bool = False,
        dtype: str = "f32",
        overlap_path: Optional[str] = None,
        left_difference_path: Optional[str] = None,
        right_difference_path: Optional[str] = None,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new HyperSketching model.

        Parameters
        --------------------------
        number_of_hops: int = 2
            The number of hops for the Sketches.
        precision: int = 6
            The precision of the HyperLogLog counters.
            The supported values range from 4 to 16.
        bits: int = 5
            The number of bits of the HyperLogLog counters.
            The supported values range from 4 to 6.
        include_node_types: bool = False,
            Whether to include node types in the sketches.
        include_edge_types: bool = False,
            Whether to include edge types in the sketches.
        include_edge_ids: bool = False,
            Whether to include edge ids in the sketches.
        include_node_ids: bool = True,
            Whether to include node ids in the sketches.
        include_selfloops: bool = True,
            Whether to include selfloops in the sketches.
        include_typed_graphlets: bool = False,
            Whether to include typed graphlets in the sketches.
        normalize_by_symmetric_laplacian: bool = True,
            Whether to normalize the sketches by the symmetric laplacian.
        concatenate_features: bool = False,
            Whether to concatenate the normalized and non-normalized features.
        dtype: str = "f32",
            The type of the features.
        overlap_path: Optional[str] = None,
            The path to the overlap file.
            This will be the position where, if provided, we will MMAP
            the overlap numpy array.
        left_difference_path: Optional[str] = None,
            The path to the left difference file.
            This will be the position where, if provided, we will MMAP
            the left difference numpy array.
        right_difference_path: Optional[str] = None,
            The path to the right difference file.
            This will be the position where, if provided, we will MMAP
            the right difference numpy array.
        ring_bell: bool = False,
            Whether to ring the bell when the sketches are ready.
        enable_cache: bool = False,
            Whether to enable caching of the sketches.
        """
        self._kwargs = dict(
            number_of_hops=number_of_hops,
            precision=precision,
            bits=bits,
            include_node_types=include_node_types,
            include_edge_types=include_edge_types,
            include_edge_ids=include_edge_ids,
            include_node_ids=include_node_ids,
            include_selfloops=include_selfloops,
            include_typed_graphlets=include_typed_graphlets,
            normalize_by_symmetric_laplacian=normalize_by_symmetric_laplacian,
            concatenate_features=concatenate_features,
            dtype=dtype,
        )

        self._overlap_path=overlap_path
        self._left_difference_path=left_difference_path
        self._right_difference_path=right_difference_path


        self._model = models.HyperSketching(
            **self._kwargs
        )

        self._fitting_was_executed = False

        super().__init__(
            enable_cache=enable_cache,
            ring_bell=ring_bell,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            **self._kwargs,
            overlap_path=self._overlap_path,
            left_difference_path=self._left_difference_path,
            right_difference_path=self._right_difference_path,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            number_of_hops=2,
            precision=4,
            bits=4,
        )
    
    def is_fit(self) -> bool:
        """Return whether the model was fit."""
        return self._fitting_was_executed
    
    def fit(
        self,
        graph: Graph,
    ):
        """Fit the model on the provided graph.

        Parameters
        -------------------
        graph: Graph,
            The graph to fit the model on.
        """
        self._fitting_was_executed = True
        self._model.fit(graph)
        
    def get_bits(self):
        """Return the number of bits used for the HyperLogLog counters."""
        return self._model.get_bits()
    
    def get_precision(self):
        """Return the precision used for the HyperLogLog counters."""
        return self._model.get_precision()
        
    def get_number_of_hops(self):
        """Return the number of hops used for the sketches."""
        return self._model.get_number_of_hops()
    
    @classmethod
    def get_feature_dictionary_keys(cls) -> List[str]:
        """Return the list of keys to be used in the feature dictionary."""
        return [
            "overlap",
            "left_difference",
            "right_difference",
        ]
    
    def get_feature_dictionary_shapes(self) -> Dict[str, List[int]]:
        """Return the dictionary of shapes to be used in the feature dictionary."""
        factor = 2 if self.get_concatenate_features() else 1
        return dict(
            overlap=[factor * self.get_number_of_hops()**2],
            left_difference=[factor * self.get_number_of_hops()],
            right_difference=[factor * self.get_number_of_hops()],
        )
    
    def get_difference_cardinalities_from_node_ids(self, src: int, dst: int)-> np.ndarray:
        """Return the cardinalities of the differences between the sketches of the two nodes.
        
        Parameters
        -------------------
        src: int,
            The source node id.
        dst: int,
            The destination node id.

        Returns
        -------------------
        The cardinalities of the differences between the sketches of the two nodes.
        This is a numpy array of shape (self.get_number_of_hops(), ).

        Raises
        -------------------
        ValueError,
            If the provided node ids are not in the graph.
            If the model was not fitted.
        """
        return self._model.get_difference_cardinalities_from_node_ids(src, dst)
    
    def get_overlap_cardinalities_from_node_ids(self, src: int, dst: int) -> np.ndarray:
        """Return the cardinalities of the overlaps between the sketches of the two nodes.

        Parameters
        -------------------
        src: int,
            The source node id.
        dst: int,
            The destination node id.

        Returns
        -------------------
        The cardinalities of the overlaps between the sketches of the two nodes.
        This is a numpy array of shape (self.get_number_of_hops(), self.get_number_of_hops()).

        Raises
        -------------------
        ValueError,
            If the provided node ids are not in the graph.
            If the model was not fitted.
        """
        return self._model.get_overlap_cardinalities_from_node_ids(src, dst)

    def get_normalize_by_symmetric_laplacian(self) -> bool:
        """Return whether the sketches are normalized by the symmetric laplacian."""
        return self._model.get_normalize_by_symmetric_laplacian()
    
    def get_concatenate_features(self) -> bool:
        """Return whether the features are concatenated."""
        return self._model.get_concatenate_features()

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return edge sketches.
        
        Parameters
        -------------------
        graph: Graph,
            The graph to fit the model on.
        return_dataframe: bool = True,
            Whether to return the results as pandas dataframes.
        """
        if not self._fitting_was_executed:
            self.fit(graph)
        (overlaps, left_difference, right_difference) = self._model.get_sketching_for_all_edges(
            graph,
            support=graph,
            overlap_path=self._overlap_path,
            left_difference_path=self._left_difference_path,
            right_difference_path=self._right_difference_path,
        )
        if return_dataframe:
            factor = 2 if self.get_concatenate_features() else 0
            extra_columns = ["N"] if self.get_concatenate_features() else []
            overlaps = pd.DataFrame(
                overlaps.reshape(-1, factor * self.get_number_of_hops()**2),
                index=graph.get_edge_node_names(),
                columns=[
                    "{column}_{i}{j}".format(column=column, i=i, j=j)
                    for column in ["A"] + extra_columns
                    for i in range(self.get_number_of_hops())
                    for j in range(self.get_number_of_hops())
                ]
            )
            left_difference = pd.DataFrame(
                left_difference.reshape(-1, factor * self.get_number_of_hops()),
                index=graph.get_edge_node_names(),
                columns=[
                    "{column}_{i}".format(column=column, i=i)
                    for column in ["L"] + extra_columns
                    for i in range(self.get_number_of_hops())
                ]
            )
            right_difference = pd.DataFrame(
                right_difference.reshape(-1, factor * self.get_number_of_hops()),
                index=graph.get_edge_node_names(),
                columns=[
                    "{column}_{i}".format(column=column, i=i)
                    for column in ["R"] + extra_columns
                    for i in range(self.get_number_of_hops())
                ]
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            edge_embeddings=[overlaps, left_difference, right_difference],
        )
    
    def get_sketching_from_edge_node_ids(
        self,
        support: Graph,
        sources: np.ndarray,
        destinations: np.ndarray,
        overlap_path: Optional[str] = None,
        left_difference_path: Optional[str] = None,
        right_difference_path: Optional[str] = None,
    ) -> Tuple[np.ndarray]:
        """Return the sketches for the provided edges.
        
        Parameters
        -------------------
        support: Graph,
            The graph from which we extract the node degrees if the
            laplacian normalization is enabled. Be advised that this
            graph should, in most cases, be the same as the one used
            to fit the model.
        sources: np.ndarray,
            The source node ids.
        destinations: np.ndarray,
            The destination node ids.
        overlap_path: Optional[str] = None,
            The path to the overlap file.
            If an overlap path was provided in the constructor and this
            parameter is None, then the overlap will be loaded from the
            file provided in the constructor.
            This will be the position where, if provided, we will MMAP
            the overlap numpy array.
        left_difference_path: Optional[str] = None,
            The path to the left difference file.
            If a left difference path was provided in the constructor and this
            parameter is None, then the left difference will be loaded from the
            file provided in the constructor.
            This will be the position where, if provided, we will MMAP
            the left difference numpy array.
        right_difference_path: Optional[str] = None,
            The path to the right difference file.
            If a right difference path was provided in the constructor and this
            parameter is None, then the right difference will be loaded from the
            file provided in the constructor.
            This will be the position where, if provided, we will MMAP
            the right difference numpy array.

        Returns
        -------------------
        The sketches for the provided edges.

        Raises
        -------------------
        ValueError,
            If the provided node ids are not in the graph.
            If the model was not fitted.
        """
        if not self._fitting_was_executed:
            raise ValueError(
                "The model was not fitted."
            )

        if overlap_path is None:
            overlap_path = self._overlap_path

        if left_difference_path is None:
            left_difference_path = self._left_difference_path
        
        if right_difference_path is None:
            right_difference_path = self._right_difference_path

        return self._model.get_sketching_from_edge_node_ids(
            support,
            sources,
            destinations,
            overlap_path=overlap_path,
            left_difference_path=left_difference_path,
            right_difference_path=right_difference_path,
        )
    
    def get_edge_feature_from_edge_node_ids(
        self,
        support: Graph,
        sources: np.ndarray,
        destinations: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return the edge feature for the given edge.
        
        Parameters
        -----------------------
        support: Graph,
            The graph to use as base for the topological metrics.
        sources: np.ndarray,
            The source node ids.
        destinations: np.ndarray,
            The destination node ids.
        """
        overlap, left_difference, right_difference = self.get_sketching_from_edge_node_ids(
            support,
            sources,
            destinations,
        )
        return dict(
            overlap=overlap,
            left_difference=left_difference,
            right_difference=right_difference,
        )
    
    def get_edge_feature_from_graph(self, graph: Graph, support: Graph) -> Dict[str, np.ndarray]:
        """Return the edge feature for the given graph.
        
        Parameters
        -----------------------
        graph: Graph,
            The graph to use as base for the topological metrics.
        support: Graph,
            The graph to use as base for the topological metrics.
        """
        if not self._fitting_was_executed:
            raise ValueError(
                "The model was not fitted."
            )
        overlap, left_difference, right_difference = self._model.get_sketching_for_all_edges(
            graph,
            support=support,
            overlap_path=self._overlap_path,
            left_difference_path=self._left_difference_path,
            right_difference_path=self._right_difference_path,
        )

        # A small debug assert to ensure the APIs are not broken.
        for feature in (overlap, left_difference, right_difference):
            assert feature.shape[0] == graph.get_number_of_edges()

        return dict(
            overlap=overlap,
            left_difference=left_difference,
            right_difference=right_difference,
        )

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the feature names."""
        return cls.model_name()
    
    @classmethod
    def task_name(cls) -> str:
        return "Edge Embedding"

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "HyperSketching"
    
    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    @classmethod
    def requires_node_types(cls) -> bool:
        """Returns whether the model requires node types."""
        return False
    
    def is_using_node_types(self) -> bool:
        """Returns whether the model is using node types."""
        return self._kwargs["include_node_types"]

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Returns whether the model requires edge types."""
        return False
    
    def is_using_edge_types(self) -> bool:
        """Returns whether the model is using edge types."""
        return self._kwargs["include_edge_types"]
    
    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False
    
    def clone(self) -> "Self":
        """Return a fresh clone of the model."""
        return HyperSketching(**self.parameters())
    
    @classmethod
    def load(cls, path: str) -> "Self":
        """Load a saved version of the model from the provided path.

        Parameters
        -------------------
        path: str
            Path from where to load the model.
        """
        data = compress_json.load(path)
        model = HyperSketching(**data["parameters"])
        model._model = models.HyperSketching.loads(
            json.dumps(data["inner_model"])
        )
        for key, value in data["metadata"].items():
            model.__setattr__(key, value)
        return model

    def dumps(self) -> Dict[str, Any]:
        """Dumps the current model as dictionary."""
        return dict(
            parameters=self.parameters(),
            inner_model=json.loads(self._model.dumps()),
            metadata=dict(
                _fitting_was_executed=self._fitting_was_executed
            )
        )

    def dump(self, path: str):
        """Dump the current model at the provided path.

        Parameters
        -------------------
        path: str
            Path from where to dump the model.
        """
        compress_json.dump(
            self.dumps(),
            path
        )
