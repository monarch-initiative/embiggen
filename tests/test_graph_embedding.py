import os.path
import pytest
from unittest import TestCase
from typing import Dict
from tqdm.auto import tqdm
from embiggen.transformers import GraphPartitionTransformer
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen.embedders import CBOW, SkipGram, GloVe
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sanitize_ml_labels import sanitize_ml_labels
import numpy as np


def report(y_true, y_pred) -> Dict:
    metrics = (roc_auc_score, average_precision_score, accuracy_score)
    return {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in metrics
    }


class TestGraphEmbedding(TestCase):
    """Test suite for the class Embiggen"""

    def setUp(self):
        self._directories = ["karate", "ppi"]
        self._paths = [
            "neg_test_edges",
            "neg_train_edges",
            "neg_validation_edges",
            "pos_train_edges",
            "pos_test_edges",
            "pos_validation_edges",
        ]

    def build_graphs(self, directory: str):
        return {
            path: EnsmallenGraph(
                edge_path=f"tests/data/{directory}/{path}.tsv",
                sources_column="subject",
                destinations_column="object",
                directed=False,
            )
            for path in self._paths
        }

    def evaluate_embedding(self, embedding: np.ndarray, graphs: Dict):
        transformer_model = GraphPartitionTransformer()
        transformer_model.fit(embedding)

        X_train, y_train = transformer_model.transform_edges(
            graphs["pos_train_edges"],
            graphs["neg_train_edges"]
        )
        X_test, _ = transformer_model.transform_edges(
            graphs["pos_test_edges"],
            graphs["neg_test_edges"]
        )
        X_validation, _ = transformer_model.transform_edges(
            graphs["pos_validation_edges"],
            graphs["neg_validation_edges"]
        )

        forest = RandomForestClassifier(max_depth=20)
        forest.fit(X_train, y_train)
        _ = forest.predict(X_train)
        _ = forest.predict(X_test)
        _ = forest.predict(X_validation)

        X_train_src, X_train_dst, y_train = transformer_model.transform_nodes(
            graphs["pos_train_edges"],
            graphs["neg_train_edges"]
        )
        X_train = np.hstack([X_train_src, X_train_dst])
        X_test_src, X_test_dst, _ = transformer_model.transform_nodes(
            graphs["pos_test_edges"],
            graphs["neg_test_edges"]
        )
        X_test = np.hstack([X_test_src, X_test_dst])
        X_validation_src, X_validation_dst, _ = transformer_model.transform_nodes(
            graphs["pos_validation_edges"],
            graphs["neg_validation_edges"]
        )
        X_validation = np.hstack([X_validation_src, X_validation_dst])

        forest = RandomForestClassifier(max_depth=20)
        forest.fit(X_train, y_train)
        _ = forest.predict(X_train)
        _ = forest.predict(X_test)
        _ = forest.predict(X_validation)

        # print({
        #     "train": report(y_train, y_train_pred),
        #     "test": report(y_test, y_test_pred),
        #     "validation": report(y_validation, y_validation_pred),
        # })

    def test_skipgram(self):
        for directory in self._directories:
            graphs = self.build_graphs(directory)
            X = tf.ragged.constant(
                graphs["pos_train_edges"].walk(10, 80))
            embedder_model = SkipGram()
            embedder_model.fit(
                X,
                graphs["pos_train_edges"].get_nodes_number(),
                epochs=2,
            )
            self.evaluate_embedding(embedder_model.embedding, graphs)

    def test_cbow(self):
        for directory in self._directories:
            graphs = self.build_graphs(directory)
            X = tf.ragged.constant(
                graphs["pos_train_edges"].walk(10, 80))
            embedder_model = CBOW()
            embedder_model.fit(
                X,
                graphs["pos_train_edges"].get_nodes_number(),
                epochs=2,
            )
            self.evaluate_embedding(embedder_model.embedding, graphs)

    def test_glove(self):
        for directory in self._directories:
            graphs = self.build_graphs(directory)
            X = tf.ragged.constant(
                graphs["pos_train_edges"].walk(10, 80))
            embedder_model = GloVe()
            embedder_model.fit(
                X,
                graphs["pos_train_edges"].get_nodes_number(),
                epochs=2,
            )
            self.evaluate_embedding(embedder_model.embedding, graphs)
