from unittest import TestCase
from embiggen import CorpusTransformer


class TestWordSequences(TestCase):

    def setUp(self):
        self._window_size = 2
        self._batch_size = 128
        self._transformer = CorpusTransformer()
        with open("./tests/data/short_bible.txt") as f:
            lines = f.readlines()
            self._transformer.fit(lines, min_count=2)
            self._tokens = self._transformer.transform(
                lines,
                min_length=self._window_size*2 + 1
            )

        self._sequence = None
