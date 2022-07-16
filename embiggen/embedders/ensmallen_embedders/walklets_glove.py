"""Module providing WalkletsGloVe model implementation."""
from typing import Optional, Dict, Any
from embiggen.embedders.ensmallen_embedders.walklets import WalkletsEnsmallen


class WalkletsGloVeEnsmallen(WalkletsEnsmallen):
    """Class providing WalkletsGloVe implemeted in Rust from Ensmallen."""

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Walklets GloVe"

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        removed = [
            "number_of_negative_samples"
        ]
        return dict(
            **{
                key: value
                for key, value in super().parameters().items()
                if key not in removed
            }
        )