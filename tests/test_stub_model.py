# Test module to check whether the stub model wrapper works as expected.

from embiggen.embedders.non_existent_embedders import NonExistentModel
import pytest


def test_stub_model():
    with pytest.raises(ModuleNotFoundError):
        _model_stub = NonExistentModel()
