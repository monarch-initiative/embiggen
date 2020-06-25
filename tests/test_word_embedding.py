import os.path
import pytest
from unittest import TestCase
from tqdm.auto import tqdm
from embiggen.transformers import TextTransformer
from embiggen.embedders import CBOW, SkipGram, GloVe


class TestWordEmbedding(TestCase):
    """Test suite for the class Embiggen"""

    def setUp(self):
        self._path = "tests/data/greatExpectations3.txt"

    def test_skipgram(self):
        text_encoder = TextTransformer(self._path)
        X, _, dictionary, _ = text_encoder.build_dataset()
        embedder_model = SkipGram()
        embedder_model.fit(X, len(dictionary), epochs=2)
        embedder_model.embedding
