"""This module provides the Graph Convolution feature preprocessor class."""
from typing import Union, List
from embiggen.utils.abstract_models.abstract_feature_preprocessor import AbstractFeaturePreprocessor
from ensmallen import models, Graph
import warnings
import numpy as np
import pandas as pd


class GraphConvolution(AbstractFeaturePreprocessor):

    def __init__(
        self,
        number_of_convolutions: int = 2,
        concatenate_features: bool = False,
        dtype: str = "f32",
        path: Union[str, List[str]] = None,
    ):
        """Create new Graph Convolution feature preprocessor.

        Parameters
        -------------------------
        number_of_convolutions: int = 2
            The number of convolutions to execute.
            By default, `2`.
        concatenate_features: bool = False
            Whether to concatenate the features at each convolution.
            By default, `false`.
        dtype: str = "f32"
            The data type to use for the convolved features.
            The supported values are `f16`, `f32` and `f64`.
            By default, `f32`.
        path: Union[str, List[str]] = None
            The path(s) were to MMAP the processed features to.
            By default, `None`.
        """
        self._kwargs = dict(
            number_of_convolutions=number_of_convolutions,
            concatenate_features=concatenate_features,
            dtype=dtype,
        )
        self._path = path
        self._graph_convolution = models.GraphConvolution(
            **self._kwargs
        )
        super().__init__()

    @classmethod
    def model_name(self) -> str:
        """Return the name of the feature preprocessor."""
        return "Graph Convolution"

    @classmethod
    def library_name(cls) -> str:
        """Return the name of the library of the feature preprocessor."""
        return "Ensmallen"

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Return whether the model requires edge types."""
        return False

    @classmethod
    def requires_node_types(cls) -> bool:
        """Return whether the model requires node types."""
        return False

    @classmethod
    def requires_edge_weights(cls) -> bool:
        """Return whether the model requires edge weights."""
        return False

    def _transform(
        self,
        node_features: List[Union[pd.DataFrame, np.ndarray]],
        support: Graph,
    ) -> Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]:
        """Transform the given node features.

        Parameters
        -------------------------
        node_features: List[Union[pd.DataFrame, np.ndarray]],
            Node feature to use to fit the transformer.
        support: Graph
            Support graph to use to transform the node features.
        """
        if self._path is not None and len(self._path) != len(node_features):
            raise ValueError(
                "The number of paths should be the same as the number of node features."
                f"Got {len(self._path)} paths and {len(node_features)} node features."
            )
        
        new_node_features = []

        for node_feature in node_features:
            if not isinstance(node_feature, np.ndarray):
                raise NotImplementedError(
                    "The node features should be provided as a numpy array. "
                    f"Got {type(node_feature)} instead."
                )
            if len(node_feature.shape) != 2:
                raise ValueError(
                    "The node features should be provided as a numpy array with "
                    "shape (number_of_nodes, number_of_features). "
                    f"Got {node_feature.shape} instead."
                )
            
            if not node_feature.data.c_contiguous:
                warnings.warn(
                    "One of the provided node features is not C contiguous. "
                    f"Specifically, the one with shape {node_feature.shape} "
                    f"and dtype {node_feature.dtype}. "
                    "This forces us to copy the data, which may be slow and "
                    "consume a lot of memory. Consider using np.ascontiguousarray "
                    "to make the array C contiguous before passing it to the "
                    "feature preprocessor."
                )
                node_feature = np.ascontiguousarray(node_feature)
            new_node_features.append(node_feature)

        node_features = new_node_features            

        if self._path is None:
            return [
                self._graph_convolution.transform(
                    support=support,
                    node_features=node_feature,
                )
                for node_feature in node_features
            ]
        return [
            self._graph_convolution.transform(
                support=support,
                node_feature=node_feature,
                path=path
            )
            for node_feature, path in zip(node_features, self._path)
        ]

    @classmethod
    def is_stocastic(cls) -> bool:
        """Return whether the model is stocastic."""
        return False
