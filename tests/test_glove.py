from embiggen import GloVe
from .test_sequences import TestSequences


class TestGloVe(TestSequences):

    def setUp(self):
        super().setUp()
        self._embedding_size = 50
        self._words, self._ctxs, self._freq = self._graph.cooccurence_matrix(
            80,
            window_size=4,
            iterations=20
        )
        self._model = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size,
        )

    def test_fit(self):
        self._model.fit(
            (self._words, self._ctxs),
            self._freq,
            epochs=2,
            verbose=False
        )

        self.assertEqual(
            self._model.embedding.shape,
            (self._graph.get_nodes_number(), self._embedding_size)
        )
