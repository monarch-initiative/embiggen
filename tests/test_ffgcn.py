from typing import List
from ensmallen import Graph
from embiggen.edge_prediction.edge_prediction_ensmallen.ffgcn import FFGCN
from embiggen.embedders.ensmallen_embedders.first_order_line import FirstOrderLINEEnsmallen
from embiggen.utils.ogb_utils import convert_ogb_link_prediction_dataset_to_grape
from embiggen.utils import EmbeddingResult
import numpy as np
import pandas as pd


def test_ffgcn_on_ogbl_ddi():
    """Testing that FFGCN works"""
    graphs: List[Graph] = convert_ogb_link_prediction_dataset_to_grape(
        "ogbl-ddi"
    )

    model: FFGCN = FFGCN(
        units=[200, 200, 200],
        number_of_steps_per_layer=1000,
        maximal_number_of_steps_without_improvement=20,
        batch_size=2**16,
        dropout=0.2,
        learning_rate=0.1,
        validation_interval=10,
        pre_training=True,
        avoid_support_collisions=False
    )

    line = FirstOrderLINEEnsmallen(verbose=True, epochs=100)
    embedding: EmbeddingResult = line.fit_transform(
        graphs["train_edge"],
    )

    model.fit(
        graph=graphs["train_edge"],
        node_features = embedding
    )

    model.dump("model.json")

    for graph, graph_name in (
        (graphs["test_edge"], "test_positive"),
        (graphs["test_edge_neg"], "test_negative"),
        (graphs["train_edge"], "train"),
    ):
        for prediction_reduce in FFGCN.available_prediction_reduce():
            model.set_prediction_reduce(prediction_reduce)

            predictions = model.predict_proba(
                graph=graph,
                support=graphs["train_edge"],
                node_features = embedding
            )

            pd.DataFrame({
                "score": predictions
            }).to_csv(f"{prediction_reduce}_{graph_name}_predictions.csv")
