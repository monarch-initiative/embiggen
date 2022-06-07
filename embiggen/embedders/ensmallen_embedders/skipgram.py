"""Module providing SkipGram model implementation."""
from typing import Optional
from embiggen.embedders.ensmallen_embedders.node2vec import Node2VecEnsmallen


class SkipGramEnsmallen(Node2VecEnsmallen):
    """Class providing SkipGram implemeted in Rust from Ensmallen."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 10,
        clipping_value: float = 6.0,
        number_of_negative_samples: int = 5,
        walk_length: int = 128,
        iterations: int = 1,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: Optional[int] = 100,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        normalize_by_degree: bool = False,
        stochastic_downsample_by_degree: Optional[bool] = False,
        normalize_learning_rate_by_degree: Optional[bool] = False,
        use_zipfian_sampling: Optional[bool] = True,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        epochs: int = 10
            Number of epochs to train the model for.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        clipping_value: float = 6.0
            Value at which we clip the dot product, mostly for numerical stability issues.
            By default, `6.0`, where the loss is already close to zero.
        number_of_negative_samples: int = 5
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        walk_length: int = 128
            Maximal length of the walks.
        iterations: int = 1
            Number of iterations of the single walks.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        return_weight: float = 1.0
            Weight on the probability of returning to the same node the walk just came from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight: float = 1.0
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        max_neighbours: Optional[int] = 100
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with high degrees.
        learning_rate: float = 0.01
            The learning rate to use to train the Node2Vec model. By default 0.01.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        normalize_by_degree: bool = False
            Whether to normalize the random walk by the node degree
            of the destination node degrees.
        stochastic_downsample_by_degree: Optional[bool] = False
            Randomly skip samples with probability proportional to the degree of the central node. By default false.
        normalize_learning_rate_by_degree: Optional[bool] = False
            Divide the learning rate by the degree of the central node. By default false.
        use_zipfian_sampling: Optional[bool] = True
            Sample negatives proportionally to their degree. By default true.
        random_state: int = 42
            The random state to reproduce the training sequence.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            model_name="SkipGram",
            embedding_size=embedding_size,
            epochs=epochs,
            clipping_value=clipping_value,
            number_of_negative_samples=number_of_negative_samples,
            walk_length=walk_length,
            iterations=iterations,
            window_size=window_size,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_edge_type_weight=change_edge_type_weight,
            change_node_type_weight=change_node_type_weight,
            max_neighbours=max_neighbours,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            normalize_by_degree=normalize_by_degree,
            stochastic_downsample_by_degree=stochastic_downsample_by_degree,
            normalize_learning_rate_by_degree=normalize_learning_rate_by_degree,
            use_zipfian_sampling=use_zipfian_sampling,
            random_state=random_state,
            enable_cache=enable_cache,
        )

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "SkipGram"
    
    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False