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
            "available": model_class.is_available()
        }
        for model_name, tasks in AbstractModel.MODELS_LIBRARY.items()
        for task_name, libraries in tasks.items()
        for library_name, model_class in libraries.items()
    ])