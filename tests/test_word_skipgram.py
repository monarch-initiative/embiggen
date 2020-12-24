"""Uni test to test that model SkipGram works properly with words sequences."""
from embiggen import SkipGram
import numpy as np
from .test_word2vec_sequences import TestWord2VecSequences


class TestWordSkipGram(TestWord2VecSequences):
    """Unit test to test that model SkipGram works properly with words sequences."""

    def setUp(self):
        """Setup object to test SkipGram model on words sequences."""
        super().setUp()
        self._embedding_size = 10
        self._model = SkipGram(
            self._transformer.vocabulary_size,
            embedding_size=self._embedding_size,
            window_size=self._window_size
        )

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        self._model.fit(
            self._sequence,
            steps_per_epoch=self._sequence.steps_per_epoch,
            epochs=2,
            verbose=False
        )

        self.assertFalse(np.isnan(self._model.embedding).any())

        self.assertEqual(
            self._model.embedding.shape,
            (self._transformer.vocabulary_size, self._embedding_size)
        )
