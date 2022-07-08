import pytest
from embiggen.utils.normalize_model_structural_parameters import normalize_model_list_parameter, normalize_model_ragged_list_parameter
from unittest import TestCase


class TestTensorFlowUtils(TestCase):

    def setUp(self):
        pass

    def test_normalize_model_list_parameter(self):
        values = normalize_model_list_parameter(
            10,
            3,
            int
        )

        self.assertEqual(values, [10, 10, 10])

        with pytest.raises(ValueError):
            normalize_model_list_parameter([], 0, int)

        with pytest.raises(ValueError):
            normalize_model_list_parameter(8.0, 5, int)

        with pytest.raises(ValueError):
            normalize_model_list_parameter([2], 5, int)

        with pytest.raises(ValueError):
            normalize_model_list_parameter([2.0], 1, int)

    def test_normalize_model_ragged_list_parameter(self):
        normalize_model_ragged_list_parameter(
            None,
            3,
            layers_per_submodule=[1, 2, 3],
            object_type=int,
            default_value=6
        )