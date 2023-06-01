"""Module providing a very abstract general features to be used in the models."""

class AbstractFeature:

    def __init__(self):
        pass

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the name of the feature."""
        raise NotImplementedError(
            "The method get_feature_name was not implemented."
        )