import os
from embiggen import GloVe
from .test_word_sequences import TestWordSequences
from ensmallen_graph import preprocessing  # pylint: disable=no-name-in-module


class TestWordGloVe(TestWordSequences):

    def setUp(self):
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
        self._model.fit(
            (self._words, self._ctxs),
            self._freq,
            epochs=2,
            verbose=False
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._transformer.vocabulary_size, self._embedding_size)
        )