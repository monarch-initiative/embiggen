"""Module providing dataframes with informations about the available models."""
import pandas as pd
from .abstract_model import AbstractModel


def get_models_dataframe() -> pd.DataFrame:
    """Returns dataframe with informations about available models."""
    return pd.DataFrame([
        {
            "model_name": model_name,
            "task_name": task_name,
            "library_name": library_name,
            "available": model_class.is_available(),
            "requires_node_types": model_class.requires_node_types(),
            "requires_edge_types": model_class.requires_edge_types(),
            "requires_edge_weights": model_class.requires_edge_weights(),
            "requires_positive_edge_weights": model_class.requires_positive_edge_weights(),
        }
        for model_name, tasks in AbstractModel.MODELS_LIBRARY.items()
        for task_name, libraries in tasks.items()
        for library_name, model_class in libraries.items()
    ])


def get_available_models_for_node_embedding() -> pd.DataFrame:
    """Returns dataframe with informations about available models for node embedding."""
    df = get_models_dataframe()
    return df[(df.task_name == "Node Embedding") & df.available]


def get_available_models_for_edge_prediction() -> pd.DataFrame:
    """Returns dataframe with informations about available models for edge prediction."""
    df = get_models_dataframe()
    return df[(df.task_name == "Edge Prediction") & df.available]


def get_available_models_for_edge_label_prediction() -> pd.DataFrame:
    """Returns dataframe with informations about available models for edge-label prediction."""
    df = get_models_dataframe()
    return df[(df.task_name == "Edge Label Prediction") & df.available]


def get_available_models_for_node_label_prediction() -> pd.DataFrame:
    """Returns dataframe with informations about available models for node-label prediction."""
    df = get_models_dataframe()
    return df[(df.task_name == "Node Label Prediction") & df.available]
