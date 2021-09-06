"""Test to validate that the model CBOW works properly with words sequences."""
from embiggen import CBOW
import numpy as np
from .test_word2vec_sequences import TestWord2VecSequences


class TestWordCBOW(TestWord2VecSequences):
    """Unit test to validate that the model CBOW works properly with words sequences."""

    def setUp(self):
        """Setup objects to run tests on CBOW model class for words sequences."""
        super().setUp()
        self._embedding_size = 10
        self._model = CBOW(
            vocabulary_size=self._transformer.vocabulary_size,
            embedding_size=self._embedding_size,
            window_size=self._window_size
        )

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        self._model.fit(
            self._sequence,
            steps_per_epoch=self._sequence.steps_per_epoch,
            epochs=2
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._transformer.vocabulary_size, self._embedding_size)
        )

        self.assertFalse(np.isnan(self._model.embedding).any())
