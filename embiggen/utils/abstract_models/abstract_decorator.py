"""Module providing a very simple abstract decorator."""
from typing import Type
from .abstract_model import AbstractModel

def abstract_class(klass: Type[AbstractModel]) -> Type[AbstractModel]:
    """Simply adds a descriptor for meta-programming and nothing else."""
    return klass