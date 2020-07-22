"""Setup standard unit test class for WordSequences."""
from unittest import TestCase

from embiggen import CorpusTransformer


class TestWordSequences(TestCase):
    """Abstract unit test class for testing models and sequences on textual corpuses."""

    def setUp(self):
        """Setting up abstract class for executing tests on words sequences."""
        self._window_size = 2
        self._batch_size = 128
        self._transformer = CorpusTransformer()
        transformer2 = CorpusTransformer(extend_synonyms=False, apply_stemming=True)
        with open("./tests/data/short_bible.txt") as bible_file:
            lines = bible_file.readlines()
            self._transformer.fit(lines, min_count=2, verbose=False)
            transformer2.fit(lines, verbose=False)
            self._transformer.get_word_id("god")

            self._tokens = self._transformer.transform(
                lines,
                min_length=self._window_size*2 + 1,
                verbose=False
            )
            self._transformer.reverse_transform(self._tokens)

        self._sequence = None
