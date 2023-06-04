"""Module providing an abstract interface for feature preprocessors.

A feature preprocessor is an object that can be used to preprocess the input features
before they are used to fit a model. This can be a very relevant step in the pipeline
of a model execution, and as such, we believe it should be part of the model itself.

This way, when the model is saved, the feature preprocessor is saved as well, and
when the model is loaded, the feature preprocessor is loaded as well.
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
import warnings
from ensmallen import Graph
from embiggen.utils.abstract_models.abstract_model import AbstractModel


class AbstractFeaturePreprocessor(AbstractModel):

    def __init__(self, random_state: Optional[int] = None):
        """Create new abstract feature preprocessor.

        Parameters
        -------------------------
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        super().__init__(random_state=random_state)

    @classmethod
    def task_name(cls) -> str:
        return "Feature Preprocessor"

    def _transform(
        self,
        support: Graph,
        node_features: List[Union[pd.DataFrame, np.ndarray]],
    ) -> Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]:
        """Transform the given node features.

        Parameters
        -------------------------
        node_features: List[Union[pd.DataFrame, np.ndarray]],
            Node feature to use to fit the transformer.
        support: Graph
            Support graph to use to transform the node features.
        """
        raise NotImplementedError(
            "The method _transform should be implemented by the child class."
        )

    def transform(
        self,
        support: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
    ) -> Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]:
        """Transform the given node features.

        Parameters
        -------------------------
        support: Graph
            Support graph to use to transform the node features.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.
        """
        assert isinstance(support, Graph)

        if not isinstance(node_features, list):
            node_features = [node_features]

        if len(node_features) == 0:
            raise ValueError(
                "You have provided an empty list of node features "
                f"to the {self.model_name()} feature preprocessor."
            )

        if len(node_features) > 1:
            warnings.warn(
                "You have provided more than one node feature to the "
                f"{self.model_name()} feature preprocessor. "
                "This feature preprocessor will be applied to each of them "
                "independently."
            )

        for feature in node_features:
            assert isinstance(feature, (pd.DataFrame, np.ndarray))

        return self._transform(support=support, node_features=node_features)
