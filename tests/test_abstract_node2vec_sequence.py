import pytest
from embiggen.sequences.abstract_node2vec_sequence import AbstractNode2VecSequence
from .test_node_sequences import TestNodeSequences


class TestAbstractNode2VecSequence(TestNodeSequences):

    def setUp(self):
        super().setUp()
        self._window_size = 4
        self._walk_length = 100
        self._batch_size = 1
        self._sequence = AbstractNode2VecSequence(
            self._graph,
            walk_length=self._walk_length,
            batch_size=self._batch_size,
            window_size=self._window_size
        )

        with pytest.raises(NotImplementedError):
            self._sequence.__getitem__(0)
