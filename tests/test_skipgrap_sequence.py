from embiggen import NodeSkipGramSequence
from .test_sequences import TestSequences


class TestSkipGramSequences(TestSequences):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._length = 100
        self._batch_size = 1
        self._sequence = NodeSkipGramSequence(
            self._graph,
            length=self._length,
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