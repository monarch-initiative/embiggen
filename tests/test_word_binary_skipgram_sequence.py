"""Unit test to verify that WordBinarySkipGramSequence behaves nominally."""
from embiggen import WordBinarySkipGramSequence

from .test_word_sequences import TestWordSequences


class TestWordBinarySkipGramSequence(TestWordSequences):
    """Unit test to verify that WordBinarySkipGramSequence behaves nominally."""

    def setUp(self):
        """Setup up objects to test WordBinarySkipGramSequence."""
        super().setUp()
        self._sequence = WordBinarySkipGramSequence(
            self._tokens,
            batch_size=self._batch_size,
            vocabulary_size=self._transformer.vocabulary_size,
            window_size=self._window_size
        )

    def test_output_shape(self):
        """Testing that shape produced by WordBinarySkipGramSequence is correct."""
        (context_vector, words_vector), _ = self._sequence[0]
        self.assertEqual(context_vector.shape[0], words_vector.shape[0])
