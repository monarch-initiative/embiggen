from embiggen.edge_prediction.edge_prediction_ensmallen.ffgcn import FFGCN
from embiggen.utils.ogb_utils import convert_ogb_link_prediction_dataset_to_grape
import numpy as np
import pandas as pd


def test_ffgcn_on_ogbl_ddi():
    """Testing that FFGCN works"""
    graphs = convert_ogb_link_prediction_dataset_to_grape(
        "ogbl-ddi"
    )

    model = FFGCN(
        units=[100, 100, 100],
        number_of_steps_per_layer=1000,
        maximal_number_of_steps_without_improvement=50,
        number_of_edges_per_mini_batch=2**20,
        pre_train=True,
        number_of_oversampling_neighbourhoods_per_node=10,
        threshold=3.0,
        skip_threshold=4.0,
        include_node_type_embedding=False,
        include_node_embedding=True,
        avoid_false_negatives=False
    )

    model.fit(
        graph=graphs["train_edge"],
    )

    for prediction_reduce in FFGCN.available_prediction_reduce():
        model.set_prediction_reduce(prediction_reduce)

        positive_predictions = model.predict_proba(
            graph=graphs["test_edge"],
        )

        negative_predictions = model.predict_proba(
            graph=graphs["test_edge_neg"],
        )

        pd.DataFrame({
            "positive": positive_predictions
        }).to_csv(f"{prediction_reduce}_positive_predictions.csv")
        pd.DataFrame({
            "negative": negative_predictions
        }).to_csv(f"{prediction_reduce}_negative_predictions.csv")
