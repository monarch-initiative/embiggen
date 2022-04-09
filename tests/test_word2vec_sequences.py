"""Test that Word2VecSequence behaves correctly."""
from embiggen import Word2VecSequence
from .test_word_sequences import TestWordSequences


class TestWord2VecSequences(TestWordSequences):
    """Unit test class to test that Word2VecSequence behaves correctly."""

    def setUp(self):
        """Setup test class."""
        super().setUp()
        self._sequence = Word2VecSequence(
            self._tokens,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        """Test that sequence output shape matches expectation."""
        context_vector, words_vector = self._sequence[0][0][0]
        self.assertEqual(context_vector.shape[0], words_vector.shape[0])
