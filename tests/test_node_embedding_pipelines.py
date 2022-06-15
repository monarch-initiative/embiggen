"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase
from embiggen.embedders import embed_graph
from embiggen import get_available_models_for_node_embedding
from tqdm.auto import tqdm

class TestNodeEmbeddingPipeline(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()

    def test_embedding_pipeline(self):
        """Test that embed pipeline works fine in SPINE."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )
        for _, row in bar:
            if row.requires_edge_weights:
                graph_name = "HomoSapiens"
                repository = "string"
            else:
                graph_name = "SpeciesTree"
                repository="string"

            bar.set_description(f"Testing embedding model {row.model_name} from library {row.library_name}")

            embed_graph(
                graph_name,
                repository=repository,
                embedding_model=row.model_name,
                library_name=row.library_name,
                verbose=False,
                smoke_test=True
            )

    def test_ensmallen_skipgram(self):
        """Test that embed pipeline works fine in SPINE."""
        graph_name = "SpeciesTree"
        repository="string"

        embed_graph(
            graph_name,
            repository=repository,
            embedding_model="SkipGram",
            verbose=False,
            smoke_test=True
        )