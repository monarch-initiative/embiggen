import pytest
try:
    from embiggen.embedders.tensorflow_embedders.edge_prediction_based_tensorflow_embedders import EdgePredictionBasedTensorFlowEmbedders
    from unittest import TestCase


    class EmbedderTest(EdgePredictionBasedTensorFlowEmbedders):

        def __init__(self):
            super().__init__()

        @classmethod
        def is_stocastic(cls) -> bool:
            return True

        @classmethod
        def model_name(cls) -> str:
            """Returns model name of the model."""
            return "Test"



    class TestEdgePredictionBasedTensorFlowEmbedders(TestCase):

        def setUp(self):
            pass

        def test_not_implemented_methods(self):

            non_stocastic = EmbedderTest()

            for method_name, params in (
                ("_build_edge_prediction_based_model", (None, None, None)),
            ):
                with pytest.raises(NotImplementedError):
                    non_stocastic.__getattribute__(method_name)(*params)
except ModuleNotFoundError:
    pass