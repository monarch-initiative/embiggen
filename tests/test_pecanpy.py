"""Test to ensure that the pecanpy integration works as expected."""
from ensmallen.datasets.kgobo import CIO
from embiggen.embedders.pecanpy_embedders.node2vec import Node2VecPecanPy


def test_pecanpy():
    """Test to ensure that the pecanpy integration works as expected."""
    graph = CIO()
    model = Node2VecPecanPy()
    _embedding = model.fit_transform(graph)
