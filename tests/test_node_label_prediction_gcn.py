"""Test to validate that the model GloVe works properly with graph walks."""
from embiggen import GraphConvolutionalNeuralNetwork
from unittest import TestCase
from ensmallen_graph.datasets.linqs import Cora


class TestShallowNodeLabelPredictionGCN(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()
        self._graph = Cora()

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        model = GraphConvolutionalNeuralNetwork(
            self._graph,
            node_features_number=10
        )
        self.assertEqual("GCN", model.name)
        model.summary()

        # Creating the normalized graph
        laplacian = self._graph.get_symmetric_normalized_transformed_graph().add_selfloops(weight=1.0)
        train, test = laplacian.node_label_holdout(
            0.8,
            use_stratification=False,
            random_state=42
        )

        model.fit(
            train,
            validation_graph=test,
            batch_size=256,
            epochs=2
        )
