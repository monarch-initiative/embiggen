from .graph import Graph
import tensorflow as tf  # type: ignore
from embiggen import SkipGram, Cbow, GloVeModel, CooccurrenceEncoder
from .transformers import GraphPartitionTransfomer
from typing import Tuple, Union, Dict
import numpy as np  # type: ignore
import enum

class EmbeddingMethod(enum.Enum):
    """
    Enumeraton of available embedding methods.
    """
    skipgram = 1
    cbow = 2
    glove = 3


class Embiggen:

    def __init__(self):
        """Returns new instance of Embiggen.

        Parameters
        ----------------------
        method: str = "skipgram",
            Method to use to transform the nodes embedding to edges.

        Raises
        ----------------------
        ValueError,
            If the given embedding method is not supported.
        """
        # TODO: a very long docstring showing the possible usages of Embiggen.
        self._model = None  # TODO! move the constructor of the model here!
        self._embedding = None

    def _get_embedding_model(
        self,
        data: Union[tf.Tensor, tf.RaggedTensor],
        worddict: Dict[str,int],
        reverse_worddict: Dict[int, str],
        embedding_model: str,
        epochs: int,
        embedding_size: int,
        context_window: int,
        window_size: int,
        devicetype="cpu"
    ):
        # TODO: add notes for the various parameters relative to which parameters
        # and add exceptions relative to the invalid ranges for the specific
        # parameters.
        if self._embedding_method == EmbeddingMethod.skipgram:
            return SkipGram(
                data=data,
                word2id=worddict,
                id2word=reverse_worddict,
                devicetype=devicetype,
            )
  
        elif embedding_model == EmbeddingMethod.glove:
            vocab_size = len(worddict)
            cencoder = CooccurrenceEncoder(
                data,
                window_size=window_size,
                vocab_size=vocab_size
            )
            return GloVeModel()
        else:
            # should never happen
            raise ValueError("Unrecognized EmbeddingModel type")

    def fit(
        self,
        data: Union[tf.Tensor, tf.RaggedTensor],
        worddict: Dict[str,int],
        reverse_worddict: Dict[int, str],
        embedding_method: str = "skipgram", 
        edge_creation="hadamard",
        learning_rate: float = 0.05,
        batch_size: int = 128,
        number_negative_samples: int = 7,
        samples_per_window=2,
        epochs: int = 10,
        embedding_size: int = 200,
        context_window: int = 2,
        window_size: int = 2,
        min_occurrences: int = 2,
        scaling_factor: float = 0.75, 
        cooccurrence_cap: int =100,
        devicetype="cpu",
        callbacks: Tuple = ()
    ):
        """Fit model using input data (Tensors dervied from a Graph or a text).

        #TODO: add notes for the various parameters relative to which parameters
        # are best in which range! Add links to papers if any are relevant!

        Parameters
        -----------------

        Raises
        -----------------
        ValueError,
            If given embedding model must be 'cbow', 'skipgram' or 'glove'.

        """
        if not isinstance(embedding_method, str):
                raise TypeError("embedding_method must be a string")
        method = embedding_method.lower()
        if method == 'skipgram':
            self._embedding_method = EmbeddingMethod.skipgram
        elif method == 'cbow':
            self._embedding_method = EmbeddingMethod.cbow
        elif method == 'glove':
            self._embedding_method = EmbeddingMethod.glove
        else:
            raise ValueError("embedding method must be one of skipgram, cbow, glove")
        if not isinstance(edge_creation, str):
            raise TypeError("edge_creation must be a string")
        if not edge_creation in {"hadamard"}:
            raise TypeError("TODO ADD OTHER EDGE CREATION METHODS")
        self._transformer = GraphPartitionTransfomer(method=edge_creation)

        if self._embedding_method == EmbeddingMethod.skipgram:
            self._model = SkipGram(
                data=data,
                word2id=worddict,
                id2word=reverse_worddict,
                devicetype=devicetype,
            )
            self._model.fit(data=data, 
                        worddict=worddict, 
                        reverse_worddict=reverse_worddict, 
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        embedding_size=embedding_size,
                        context_window=context_window,
                        number_negative_samples=number_negative_samples,
                        samples_per_window=samples_per_window,
                        callbacks=callbacks)
        elif self._embedding_method == EmbeddingMethod.cbow:
            self._model = Cbow(devicetype=devicetype)
            self._model.fit(data=data, 
                        word2id=worddict, 
                        id2word=reverse_worddict, 
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        embedding_size=embedding_size,
                        context_window=context_window,
                        number_negative_samples=number_negative_samples,
                        callbacks=callbacks)
        elif self._embedding_method == EmbeddingMethod.glove:
            self._model = GloVeModel()
            print("About to Glove fit")
            self._model.fit(data=data, 
                        worddict=worddict, 
                        reverse_worddict=reverse_worddict, 
                        learning_rate=learning_rate,
                        min_occurrences=min_occurrences,
                        scaling_factor=scaling_factor,
                        cooccurrence_cap=cooccurrence_cap,
                        batch_size=batch_size,
                        epochs=epochs,
                        embedding_size=embedding_size,
                        context_window=context_window,
                        callbacks=callbacks)
        # !TODO WHAT SHOULD THE TRANSFORMER DO?
        #self._transformer.fit(self._embedding)

    def transform(self, positives: Graph, negatives: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return tuple of embedded positives and negatives graph partitions.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Tuple of X and y to be used for training.
        """
        return self._transformer.transform(positives, negatives)

    def transform_nodes(self, positives: Graph, negatives: Graph) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Return triple of embedded positives and negatives graph partitions.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Triple of X for the source nodes,
        X for the destination nodes and the labels to be used for training.
        """
        return self._transformer.transform_nodes(positives, negatives)

    @property
    def embedding(self) -> np.ndarray:
        """Return computed embedding.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.

        Returns
        ---------------------
        Computed embedding.
        """
        return self._model._embedding

    def save_embedding(self, path: str):
        """Save the computed embedding to the given file.

        Parameters
        -------------------
        path: str,
            Path where to save the embedding.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.

        """
        self._model.save_embedding(path)

    def load_embedding(self, path: str):
        """Load the embedding at the given path.

        Parameters
        -------------------
        path: str,
            Path from where to load the embedding.

        Raises
        ---------------------
        ValueError,
            If the give path does not exists.
        """
        self._model.load_embedding(path)
