"""Module providing dataframes with informations about the available models."""
from typing import Type
import pandas as pd
from embiggen.utils.abstract_models.abstract_model import AbstractModel


def get_model_metadata(model_class: Type[AbstractModel]):
    """Return meetadata for the given model."""
    try:
        return {
            "model_name": model_class.model_name(),
            "task_name": model_class.task_name(),
            "library_name": model_class.library_name(),
            "available": model_class.is_available(),
            "requires_node_types": model_class.requires_node_types(),
            "can_use_node_types": model_class.requires_node_types() or model_class.can_use_node_types(),
            "requires_edge_types": model_class.requires_edge_types(),
            "can_use_edge_types": model_class.requires_edge_types() or model_class.can_use_edge_types(),
            "requires_edge_weights": model_class.requires_edge_weights(),
            "can_use_edge_weights": model_class.requires_edge_weights() or model_class.can_use_edge_weights(),
            "requires_positive_edge_weights": model_class.requires_positive_edge_weights(),
        }
    except NotImplementedError as e:
        raise NotImplementedError(
            "Some of the mandatory static methods were not "
            f"implemented in model class {model_class.__name__}. "
            f"The previous exception was: {str(e)}"
        )


def get_models_dataframe() -> pd.DataFrame:
    """Returns dataframe with informations about available models."""
    return pd.DataFrame([
        get_model_metadata(model_class)
        for tasks in AbstractModel.MODELS_LIBRARY.values()
        for libraries in tasks.values()
        for model_class in libraries.values()
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
