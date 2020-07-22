"""Test to validate that the model CBOW works properly with graph walks."""
import os
from embiggen import CBOW
from .test_node2vec_sequence import TestNode2VecSequence


class TestNodeCBOW(TestNode2VecSequence):

    def setUp(self):
        super().setUp()
        self._embedding_size = 50
        self._model = CBOW(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        )
        self.assertEqual("CBOW", self._model.name)
        self._model.summary()

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
            (self._graph.get_nodes_number(), self._embedding_size)
        )

        self._model.save_weights(self._weights_path)
        self._model.load_weights(self._weights_path)
        self._model.save_embedding(self._embedding_path, self._graph.nodes_reverse_mapping)
        os.remove(self._weights_path)
