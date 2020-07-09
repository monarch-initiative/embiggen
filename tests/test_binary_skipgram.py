import os
from embiggen import BinarySkipGram
from .test_binary_skipgrap_sequence import TestBinarySkipGramSequences


class TestBinarySkipGram(TestBinarySkipGramSequences):

    def setUp(self):
        super().setUp()
        self._embedding_size = 50
        self._model = BinarySkipGram(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        )
        self.assertEqual("BinarySkipGram", self._model.name)
        self._model.summary()

    def test_fit(self):
        self._model.fit(
            self._sequence,
            steps_per_epoch=self._sequence.steps_per_epoch,
            epochs=2,
            verbose=False
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._graph.get_nodes_number(), self._embedding_size)
        )

        self._model.save_weights(self._weights_path)
        self._model.load_weights(self._weights_path)
        os.remove(self._weights_path)