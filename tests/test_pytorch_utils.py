import pytest
try:
    from embiggen.utils.pytorch_utils import validate_torch_device
    from unittest import TestCase


    class TestPytorchUtils(TestCase):

        def setUp(self):
            pass

        def test_pytorch_devices(self):
            self.assertEqual(validate_torch_device("cpu"), "cpu")

            with pytest.raises(ValueError):
                validate_torch_device("huhu")

            try:
                import torch
                if not torch.cuda.is_available():
                    self.assertEqual(validate_torch_device("auto"), "cpu")
                    with pytest.raises(ValueError):
                        validate_torch_device("cuda")
            except ModuleNotFoundError:
                validate_torch_device("cuda")
except ModuleNotFoundError:
    pass