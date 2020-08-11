from embiggen import Node2VecSequence
from .test_abstract_node2vec_sequence import TestAbstractNode2VecSequence


class TestNode2VecSequence(TestAbstractNode2VecSequence):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._walk_length = 100
        self._batch_size = 1
        self._sequence = Node2VecSequence(
            self._graph,
            walk_length=self._walk_length,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        (context_vector, words_vector), _ = self._sequence[0]
        self.assertEqual(
            context_vector.shape,
            ((self._walk_length - self._window_size*2 + 1)*self._batch_size, self._window_size*2))
        self.assertEqual(context_vector.shape[0], words_vector.shape[0])

    def test_nodes_range(self):
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][0].flatten()))
        self.assertTrue(self.check_nodes_range(self._sequence[0][0][1]))
        self.assertTrue(self._sequence[0][1] is None)
