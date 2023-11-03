"""Test to ensure that the fastnode2vec integration works as expected."""
from ensmallen.datasets.kgobo import CIO
from embiggen.embedders.fastnode2vec_embedders.node2vec import Node2VecFastNode2Vec


def test_fastnode2vec():
    """Test to ensure that the fastnode2vec integration works as expected."""
    graph = CIO()
    model = Node2VecFastNode2Vec()
    _embedding = model.fit_transform(graph)
