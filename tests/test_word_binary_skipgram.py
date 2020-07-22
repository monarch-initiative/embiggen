"""Test to validate that the BinarySkipGram model works properly on words sequences."""
from embiggen import BinarySkipGram

from .test_word_binary_skipgram_sequence import TestWordBinarySkipGramSequence


class TestWordBinarySkipGram(TestWordBinarySkipGramSequence):
    """Unit test for verifying that BinarySkipGram model works correctly on words sequences."""

    def setUp(self):
        """Setting up objects to test that BinarySkipGram model class works correctly."""
        super().setUp()
        self._embedding_size = 10
        self._model = BinarySkipGram(
            self._transformer.vocabulary_size,
            embedding_size=self._embedding_size
        )

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        self._model.fit(
            self._sequence,
            steps_per_epoch=self._sequence.steps_per_epoch,
            epochs=2,
            verbose=False
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._transformer.vocabulary_size, self._embedding_size)
        )
