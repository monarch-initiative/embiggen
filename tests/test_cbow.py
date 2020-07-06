from embiggen import CBOW
from .test_cbow_sequence import TestCBOWSequences


class TestCBOW(TestCBOWSequences):

    def setUp(self):
        super().setUp()
        self._embedding_size = 50
        self._model = CBOW(
            vocabulary_size=self._graph.get_nodes_number(),
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
            (self._graph.get_nodes_number(), self._embedding_size)
        )
