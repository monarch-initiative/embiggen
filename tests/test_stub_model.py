# Test module to check whether the stub model wrapper works as expected.

from embiggen.embedders.non_existent_embedders import NonExistentModel


def test_stub_model():
    model_stub = NonExistentModel()
