from embiggen import Word2VecSequence, CBOW
from .test_word2vec_sequences import TestWord2VecSequences


class TestWordCBOWSequences(TestWord2VecSequences):

    def setUp(self):
        super().setUp()
        self._embedding_size = 10
        self._model = CBOW(
            self._transformer.vocabulary_size,
            embedding_size=self._embedding_size,
            window_size=self._window_size
        )

    def test_fit(self):
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