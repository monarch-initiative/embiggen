"""GCN model for edge-labek prediction."""
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
from ensmallen import Graph
from embiggen.utils.abstract_edge_gcn import AbstractEdgeGCN
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from embiggen.sequences.tensorflow_sequences import GCNEdgeLabelPredictionTrainingSequence


class GCNEdgeLabelPrediction(AbstractEdgeGCN, AbstractEdgeLabelPredictionModel):
    """GCN model for edge-label prediction."""

    def _get_model_training_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> GCNEdgeLabelPredictionTrainingSequence:
        """Returns training input tuple."""
        return GCNEdgeLabelPredictionTrainingSequence(
            graph=graph,
            support=support,
            kernel=self._graph_to_kernel(support),
            node_features=node_features,
            return_node_ids=self._use_node_embedding,
            return_node_types=self.is_using_node_types(),
            node_type_features=node_type_features,
            use_edge_metrics=self._use_edge_metrics,
            edge_features=edge_features,
        )

    def _get_class_weights(self, graph: Graph) -> Dict[int, float]:
        """Returns dictionary with class weights."""
        number_of_directed_edges = graph.get_number_of_directed_edges()
        edge_types_number = graph.get_edge_types_number()
        return {
            edge_type_id: number_of_directed_edges / count / edge_types_number
            for edge_type_id, count in graph.get_edge_type_id_counts_hashmap().items()
        }

    @staticmethod
    def model_name() -> str:
        return "GCN"

    def get_output_classes(self, graph:Graph) ->int:
        """Returns number of output classes."""
        return graph.get_edge_types_number()