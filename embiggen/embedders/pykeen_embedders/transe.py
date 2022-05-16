"""Submodule providing wrapper for PyKeen's TransE model."""
from pykeen.models import TransE
from .entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class TransEPyKeen(EntityRelationEmbeddingModelPyKeen):

    @staticmethod
    def model_name() -> str:
        """Return name of the model."""
        return "TransE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> TransE:
        """Build new TransE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return TransE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            scoring_fct_norm=self._scoring_fct_norm
        )
