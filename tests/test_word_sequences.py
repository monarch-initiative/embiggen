"""Setup standard unit test class for WordSequences."""
from unittest import TestCase
import pytest
import numpy as np
from embiggen import CorpusTransformer


class TestWordSequences(TestCase):
    """Abstract unit test class for testing models and sequences on textual corpuses."""

    def setUp(self):
        """Setting up abstract class for executing tests on words sequences."""
        self._window_size = 2
        self._batch_size = 128
        with pytest.raises(ValueError):
            CorpusTransformer(
                verbose=False,
                tokenizer_method="unsupported"
            )
        self._transformer = CorpusTransformer(
            verbose=False,
            min_sequence_length=self._window_size*2+1
        )
        with open("./tests/data/short_bible.txt") as bible_file:
            lines = bible_file.readlines()
            self._transformer.fit(lines)
            self._transformer.get_word_id("god")
            self._tokens = self._transformer.transform(lines)
            self._transformer.reverse_transform(self._tokens)

        self._sequence = None

        with pytest.raises(ValueError):
            self._transformer.fit([
                "test",
                "with",
                np.nan
            ])
        with pytest.raises(ValueError):
            self._transformer.transform([
                "test",
                "with",
                np.nan
            ])
        with pytest.raises(ValueError):
            self._transformer.fit([
                "test",
                "with",
                10
            ])
        with pytest.raises(ValueError):
            self._transformer.transform([
                "test",
                "with",
                10
            ])
