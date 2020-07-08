from embiggen import NodeCBOWSequence
from .test_sequences import TestSequences


class TestCBOWSequences(TestSequences):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._length = 100
        self._batch_size = 1
        self._sequence = NodeCBOWSequence(
            self._graph,
            length=self._length,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        (context_vector, words_vector), _ = self._sequence[0]
        self.assertEqual(
            context_vector.shape,
            ((self._length - self._window_size*2 + 1)*self._batch_size, self._window_size*2))
        self.assertEqual(context_vector.shape[0], words_vector.shape[0])

    def test_nodes_range(self):
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][0].flatten()))
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][1]))
        self.assertTrue(self._sequence[0][1] is None)
