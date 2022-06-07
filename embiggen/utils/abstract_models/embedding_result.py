"""Subclass providing EmbeddingResult object."""
from typing import List, Union, Optional, Dict
import pandas as pd
import numpy as np


class EmbeddingResult:

    def __init__(
        self,
        embedding_method_name: str,
        node_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_type_embeddings: Optional[Union[pd.DataFrame,
                                             np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Create new Embedding Result.

        Parameters
        ---------------------------
        embedding_method_name: str
            The embedding algorithm used.
        node_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node embedding(s).
            Some algorithms return multiple node embedding.
        edge_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge embedding(s).
            Some algorithms return multiple edge embedding.
        node_type_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type embedding(s).
            Some algorithms return multiple node type embedding.
        edge_type_embeddings: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge type embedding(s).
            Some algorithms return multiple edge type embedding.
        """
        if node_embeddings is not None and not isinstance(node_embeddings, list):
            node_embeddings = [node_embeddings]

        if edge_embeddings is not None and not isinstance(edge_embeddings, list):
            edge_embeddings = [edge_embeddings]

        if node_type_embeddings is not None and not isinstance(node_type_embeddings, list):
            node_type_embeddings = [node_type_embeddings]

        if edge_type_embeddings is not None and not isinstance(edge_type_embeddings, list):
            edge_type_embeddings = [edge_type_embeddings]

        for embedding_list, embedding_list_name in (
            (node_embeddings, "node embedding"),
            (edge_embeddings, "edge embedding"),
            (node_type_embeddings, "node type embedding"),
            (edge_type_embeddings, "node edge embedding"),
        ):
            if embedding_list is None:
                continue
            for embedding in embedding_list:
                if not isinstance(embedding, (np.ndarray, pd.DataFrame)):
                    raise ValueError(
                        f"One of the provided {embedding_list_name} "
                        f"computed with the {embedding_method_name} method is neither a "
                        f"numpy array or a pandas DataFrame, but a `{type(embedding)}` object."
                    )
                if embedding.shape[0] == 0:
                    raise ValueError(
                        "One of the provided {embedding_list_name} "
                        f"computed with the {embedding_method_name} method "
                        "is empty."
                    )

                if isinstance(embedding, pd.DataFrame):
                    numpy_embedding = embedding.to_numpy()
                else:
                    numpy_embedding = embedding

                if np.isnan(numpy_embedding).any():
                    raise ValueError(
                        f"One of the provided {embedding_list_name} "
                        f"computed with the {embedding_method_name} method "
                        "contains NaN values."
                    )

        self._embedding_method_name = embedding_method_name
        self._node_embeddings = node_embeddings
        self._edge_embeddings = edge_embeddings
        self._node_type_embeddings = node_type_embeddings
        self._edge_type_embeddings = edge_type_embeddings

    def get_all_node_embedding(self) -> List[Union[pd.DataFrame, np.ndarray]]:
        """Return a list with all the computed node embedding."""
        if self._node_embeddings is None:
            raise ValueError(
                "The node embedding were requested but they "
                f"were not computed by the {self._embedding_method_name} method."
            )
        return self._node_embeddings

    def get_all_edge_embedding(self) -> List[Union[pd.DataFrame, np.ndarray]]:
        """Return a list with all the computed edge embedding."""
        if self._edge_embeddings is None:
            raise ValueError(
                "The edge embedding were requested but they "
                f"were not computed by the {self._embedding_method_name} method."
            )
        return self._edge_embeddings

    def get_all_node_type_embeddings(self) -> List[Union[pd.DataFrame, np.ndarray]]:
        """Return a list with all the computed node type embedding."""
        if self._node_type_embeddings is None:
            raise ValueError(
                "The node types embedding were requested but they "
                f"were not computed by the {self._embedding_method_name} method."
            )
        return self._node_type_embeddings

    def get_all_edge_type_embeddings(self) -> List[Union[pd.DataFrame, np.ndarray]]:
        """Return a list with all the computed edge type embedding."""
        if self._edge_type_embeddings is None:
            raise ValueError(
                "The edge types embedding were requested but they "
                f"were not computed by the {self._embedding_method_name} method."
            )
        return self._edge_type_embeddings

    def get_node_embedding_from_index(self, index: int) -> Union[pd.DataFrame, np.ndarray]:
        """Return a computed node embedding curresponding to the provided index.

        Parameters
        ----------------
        index: int
            The index of the node embedding to return.

        Raises
        ----------------
        IndexError
            If the provided index is higher than the number of available embeddings.
        """
        if index >= len(self._node_embeddings):
            raise ValueError(
                f"The node embedding computed with the {self._embedding_method_name} method "
                f"are {len(self._node_embeddings)}, but you requested the embedding "
                f"in position {index}."
            )
        return self._node_embeddings[index]

    def get_edge_embedding_from_index(self, index: int) -> Union[pd.DataFrame, np.ndarray]:
        """Return a computed edge embedding curresponding to the provided index.

        Parameters
        ----------------
        index: int
            The index of the edge embedding to return.

        Raises
        ----------------
        IndexError
            If the provided index is higher than the number of available embeddings.
        """
        if index >= len(self._edge_embeddings):
            raise ValueError(
                f"The edge embedding computed with the {self._embedding_method_name} method "
                f"are {len(self._edge_embeddings)}, but you requested the embedding "
                f"in position {index}."
            )
        return self._edge_embeddings[index]

    def get_node_type_embedding_from_index(self, index: int) -> Union[pd.DataFrame, np.ndarray]:
        """Return a computed node type embedding curresponding to the provided index.

        Parameters
        ----------------
        index: int
            The index of the node type embedding to return.

        Raises
        ----------------
        IndexError
            If the provided index is higher than the number of available embeddings.
        """
        node_types_embedding = self.get_all_node_type_embeddings()
        if index >= len(node_types_embedding):
            raise ValueError(
                f"The node type embedding computed with the {self._embedding_method_name} method "
                f"are {len(node_types_embedding)}, but you requested the embedding "
                f"in position {index}."
            )
        return node_types_embedding[index]

    def get_edge_type_embedding_from_index(self, index: int) -> Union[pd.DataFrame, np.ndarray]:
        """Return a computed edge type embedding curresponding to the provided index.

        Parameters
        ----------------
        index: int
            The index of the edge type embedding to return.

        Raises
        ----------------
        IndexError
            If the provided index is higher than the number of available embeddings.
        """
        edge_types_embedding = self.get_all_edge_type_embeddings()
        if index >= len(edge_types_embedding):
            raise ValueError(
                f"The edge type embedding computed with the {self._embedding_method_name} method "
                f"are {len(edge_types_embedding)}, but you requested the embedding "
                f"in position {index}."
            )
        return edge_types_embedding[index]

    @staticmethod
    def load(cached_embedding_result: Dict[str, Union[str, List[Union[np.ndarray, pd.DataFrame]]]]) -> "EmbeddingResult":
        """Return restored embedding result."""
        return EmbeddingResult(**cached_embedding_result)

    def dump(self) -> Dict[str, Union["CachableList", "CachableValue"]]:
        """Method to cache the embedding result object."""
        return {
            "embedding_method_name": self._embedding_method_name,
            "node_embeddings": self._node_embeddings,
            "edge_embeddings": self._edge_embeddings,
            "node_type_embeddings": self._node_type_embeddings,
            "edge_type_embeddings": self._edge_type_embeddings,
        }
