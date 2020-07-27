import pytest
from embiggen import LinkPredictionSequence, GloVe
from .test_node_sequences import TestNodeSequences


class TestLinkPredictionSequence(TestNodeSequences):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._length = 100
        self._batch_size = 128
        self._embedding_size = 100
        self._sequence = LinkPredictionSequence(
            self._graph,
            embedding=GloVe(
                self._graph.get_nodes_number(),
                self._embedding_size
            ).embedding,
            batch_size=self._batch_size
        )

    def test_output_shape(self):
        for i in range(20):
            x, y = self._sequence[i]
            self.assertEqual(x.shape[1], self._embedding_size)
            self.assertTrue(x.shape[0] <= self._batch_size)
            self.assertEqual(x.shape[0], y.shape[0])

    def test_illegal_arguments(self):
        with pytest.raises(ValueError):
            LinkPredictionSequence(
                self._graph,
                embedding=GloVe(
                    self._graph.get_nodes_number(),
                    10
                ).embedding,
                method="not supported"
            )
