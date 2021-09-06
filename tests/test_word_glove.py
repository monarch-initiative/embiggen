"""Test to validate that the model GloVe works properly with words sequences."""
from ensmallen import preprocessing  # pylint: disable=no-name-in-module
import numpy as np
from embiggen import GloVe

from .test_word_sequences import TestWordSequences


class TestWordGloVe(TestWordSequences):
    """Unit test class for testing that GloVe model works correctly with words sequences."""

    def setUp(self):
        """Setup objects for testing that GloVe model works correctly with words sequences."""
        super().setUp()
        self._embedding_size = 50
        self._words, self._ctxs, self._freq = preprocessing.cooccurence_matrix(
            self._tokens,
            window_size=self._window_size,
            verbose=False
        )
        self._model = GloVe(
            vocabulary_size=self._transformer.vocabulary_size,
            embedding_size=self._embedding_size,
        )
        self.assertEqual("GloVe", self._model.name)
        self._model.summary()

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        self._model.fit(
            (self._words, self._ctxs),
            self._freq,
            epochs=2
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._transformer.vocabulary_size, self._embedding_size)
        )

        self.assertFalse(np.isnan(self._model.embedding).any())
