"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase

import pytest
from embiggen.embedders import embed_graph, HOPEEnsmallen
from ensmallen.datasets.kgobo import CIO
from embiggen import get_available_models_for_node_embedding
from tqdm.auto import tqdm

from embiggen.utils.abstract_models.abstract_embedding_model import AbstractEmbeddingModel

class TestNodeEmbeddingPipeline(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()

    def test_embedding_pipeline(self):
        """Test that embed pipeline works."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )
        for _, row in bar:
            if row.requires_edge_weights:
                graph_name = "Usair97"
                repository = "networkrepository"
            else:
                graph_name = "Cora"
                repository="linqs"

            bar.set_description(f"Testing {row.model_name} from {row.library_name}")

            embed_graph(
                graph_name,
                repository=repository,
                embedding_model=row.model_name,
                library_name=row.library_name,
                smoke_test=True
            )

    def test_hope_ensmallen(self):
        """Test that embed pipeline works."""
        graph_name = "CIO"
        repository="kgobo"

        with pytest.raises(ValueError):
            embed_graph(
                graph_name,
                repository=repository,
                embedding_model="HOPE",
                metric="Jaccard",
                root_node_name="Hallo!",
                embedding_size=5
            )
        
        with pytest.raises(ValueError):
            embed_graph(
                graph_name,
                repository=repository,
                embedding_model="HOPE",
                metric="Ancestors Jaccard",
                root_node_name=None,
                embedding_size=5
            )

        for metric in HOPEEnsmallen.get_available_metrics():
            if "ancestor" in metric.lower():
                root_node_name = CIO().get_node_name_from_node_id(0)
            else:
                root_node_name = None
            embed_graph(
                graph_name,
                repository=repository,
                embedding_model="HOPE",
                metric=metric,
                root_node_name=root_node_name,
                embedding_size=5
            )
    
    def test_model_recreation(self):
        df = get_available_models_for_node_embedding()

        for _, row in df.iterrows():
            model = AbstractEmbeddingModel.get_model_from_library(
                model_name=row.model_name,
                task_name="Node Embedding",
                library_name=row.library_name
            )()
            parameters = model.parameters()
            try:
                second_model = AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    task_name="Node Embedding",
                    library_name=row.library_name,
                )(**parameters)
                for key, value in second_model.parameters().items():
                    self.assertEqual(parameters[key], value)
            except Exception as e:
                raise ValueError(
                    f"Found an error in model {row.model_name} "
                    f"implemented in library {row.library_name}."
                ) from e