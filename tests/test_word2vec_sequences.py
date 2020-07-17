from embiggen import Word2VecSequence
from .test_word_sequences import TestWordSequences


class TestWord2VecSequences(TestWordSequences):

    def setUp(self):
        super().setUp()
        self._sequence = Word2VecSequence(
            self._tokens,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        (context_vector, words_vector), _ = self._sequence[0]
        self.assertEqual(context_vector.shape[0], words_vector.shape[0])
