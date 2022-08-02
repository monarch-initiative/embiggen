import pytest
from embiggen.utils import AbstractModel
from unittest import TestCase


class ClassForTestAbstractModelStocastic(AbstractModel):

    def __init__(self):
        super().__init__(42)

    @classmethod
    def is_stocastic(cls) -> bool:
        return True

    @classmethod 
    def can_use_edge_weights(cls) -> bool:
        return False

    @classmethod 
    def can_use_node_types(cls) -> bool:
        return False

    @classmethod 
    def can_use_edge_types(cls) -> bool:
        return False

        
class ClassForTestAbstractModelNonStocastic(AbstractModel):

    def __init__(self):
        super().__init__()

    @classmethod
    def is_stocastic(cls) -> bool:
        return False

    @classmethod 
    def requires_edge_weights(cls) -> bool:
        return True

    @classmethod 
    def requires_node_types(cls) -> bool:
        return True

    @classmethod 
    def requires_edge_types(cls) -> bool:
        return True


class TestAbstractModel(TestCase):

    def setUp(self):
        pass

    def test_not_implemented_methods(self):

        stocastic = ClassForTestAbstractModelStocastic()
        non_stocastic = ClassForTestAbstractModelNonStocastic()

        for method_name in (
            "smoke_test_parameters",
            "task_involves_edge_weights",
            "task_involves_topology",
            "is_topological",
            "task_involves_node_types",
            "task_involves_edge_types",
            "task_name",
            "library_name",
            "model_name",
            "clone",
            "consistent_hash",
        ):
            with pytest.raises(NotImplementedError):
                stocastic.__getattribute__(method_name)()
            with pytest.raises(NotImplementedError):
                non_stocastic.__getattribute__(method_name)()

    def test_implemented_methods(self):

        stocastic = ClassForTestAbstractModelStocastic()
        non_stocastic = ClassForTestAbstractModelNonStocastic()

        for method_name in (
            "parameters",
        ):
            stocastic.__getattribute__(method_name)()
            non_stocastic.__getattribute__(method_name)()

    def test_find_available_models(self):
        self.assertEqual(len(AbstractModel.find_available_models("Walklets SkipGram", "Node Embedding")), 2)
        self.assertEqual(len(AbstractModel.find_available_models("Walklets CBOW", "Node Embedding")), 1)

        with pytest.raises(ValueError):
            AbstractModel.find_available_models(
                "Walklets Impossible",
                "Node Embedding"
            )

        with pytest.raises(ValueError):
            AbstractModel.find_available_models(
                "Walklets Impossible",
                "Edge Parapello"
            )