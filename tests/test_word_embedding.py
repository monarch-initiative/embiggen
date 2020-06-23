import os.path
import pytest
from unittest import TestCase
from tqdm.auto import tqdm
from embiggen.utils import TextEncoder
from embiggen.embedders import CBOW, SkipGram, GloVe


class TestWordEmbedding(TestCase):
    """Test suite for the class Embiggen"""

    def setUp(self):
        self._path = "tests/data/greatExpectations3.txt"

    def test_skipgram(self):
        text_encoder = TextEncoder(self._path)
        X, _, dictionary, _ = text_encoder.build_dataset()
        embedder_model = SkipGram()
        embedder_model.fit(X, len(dictionary))
        embedder_model.embedding
