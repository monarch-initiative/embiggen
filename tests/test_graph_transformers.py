import os.path
import pytest
from unittest import TestCase
from typing import Dict
from tqdm.auto import tqdm
from embiggen.transformers import GraphPartitionTransformer


class TestGraphTransformers(TestCase):

    def test_invalid_state_graph_transform(self):
        with pytest.raises(ValueError):
            GraphPartitionTransformer(None)
        transfomer = GraphPartitionTransformer()
        with pytest.raises(ValueError):
            transfomer.transform_edges(None, None)
