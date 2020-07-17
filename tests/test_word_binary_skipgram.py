from embiggen import Word2VecSequence, BinarySkipGram
from .test_word_binary_skipgram_sequence import TestWordBinarySkipGramSequence


class TestWordBinarySkipGram(TestWordBinarySkipGramSequence):

    def setUp(self):
        super().setUp()
        self._embedding_size = 10
        self._model = BinarySkipGram(
            self._transformer.vocabulary_size,
            embedding_size=self._embedding_size
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
