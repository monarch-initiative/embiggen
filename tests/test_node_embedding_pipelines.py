"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase
from embiggen import embed_graph, get_available_models_for_node_embedding
from tqdm.auto import tqdm

class TestNodeEmbeddingPipeline(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()

    def test_embedding_pipeline(self):
        """Test that embed pipeline works fine in SPINE."""
        df = get_available_models_for_node_embedding()
        for _, row in tqdm(
            df.iterrows(),
            total=df.shape[0],
            desc="Testing embedding methods"
        ):
            if row.requires_edge_weights:
                graph_name = "HomoSapiens"
                repository = "string"
            else:
                graph_name = "Cora"
                repository="linqs"

            embed_graph(
                graph_name,
                repository=repository,
                embedding_model=row.model_name,
                library_name=row.library_name,
                verbose=False,
                embedding_size=5
            )