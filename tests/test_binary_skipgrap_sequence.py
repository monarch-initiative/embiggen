from embiggen import NodeBinarySkipGramSequence
from .test_node_sequences import TestNodeSequences


class TestBinarySkipGramSequences(TestNodeSequences):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._walk_length = 100
        self._batch_size = 1
        self._sequence = NodeBinarySkipGramSequence(
            self._graph,
            walk_length=self._walk_length,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        for i in range(20):
            (context_vector, words_vector), labels = self._sequence[i]
            self.assertEqual(context_vector.shape[0], words_vector.shape[0])
            self.assertEqual(labels.shape[0], words_vector.shape[0])

    def test_nodes_range(self):
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][0]))
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][1]))