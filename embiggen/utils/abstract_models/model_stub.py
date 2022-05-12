"""Submodule providing auto model stub for non-mandatory modules."""
from typing import Type
import inspect
from .abstract_model import AbstractModel
from ..list_formatting import format_list


def get_model_or_stub(
    module_library_name: str,
    formatted_library_name: str,
    submodule_name: str,
    model_class_name: str,
    formatted_model_name: str,
    parent_class: Type[AbstractModel]
) -> Type[AbstractModel]:
    """Returns either the class or a stub with helpful error messages.

    Parameters
    -------------------
    module_library_name: str
        Name of the library dependency to be check for availability.
    formatted_library_name: str
        The formatted name of the library for visualization pourposes.
    submodule_name: str
        Name of the submodule to load.
    model_class_name: str
        Name of the model class to load or stub.
    formatted_model_name: str
        Formatted model name to load.
    parent_class: Type[AbstractModel]
        Expected parent class of the model.
    """
    # We check that all names are actually string.
    for variable_name, variable_value in (
        ("module_library_name", module_library_name),
        ("formatted_library_name", formatted_library_name),
        ("submodule_name", submodule_name),
        ("model_class_name", model_class_name),
    ):
        if not issubclass(variable_value, str):
            raise ValueError(
                f"The provided parent class {variable_name} is not string. "
                "This is likely an implementation error, and should be "
                "reported to the Embiggen repository as an issue."
            )
        if variable_value == "":
            raise ValueError(
                f"The provided parent class {variable_name} is an empty string. "
                "This is likely an implementation error, and should be "
                "reported to the Embiggen repository as an issue."
            )
    # We check that the provided parent class is actually a subclass
    # of the AbstractModel interface.
    if not issubclass(parent_class, AbstractModel):
        raise ValueError(
            f"The provided parent class {parent_class} is not a child "
            "of AbstractModel. This is "
            "likely an implementation error, and should be "
            "reported to the Embiggen repository as an issue."
        )
    # We retrieve the context of the caller.
    frame = inspect.currentframe()
    # We identify what module or submodule is it calling from.
    current_module_name = frame.f_back.f_locals["__name__"]
    # We try to import the required class.
    try:
        # We try to retrieve the requested model class.
        model_class = getattr(
            __import__(
                f"{current_module_name}.{submodule_name}",
                fromlist=(model_class_name,)
            ),
            model_class_name
        )
        # We check that the loaded class is effectively an
        # implementation of the expected parent class.
        if not issubclass(model_class, parent_class):
            raise ValueError(
                f"The provided model class {model_class_name} is not "
                f"an implementation of {parent_class}. This is "
                "likely an implementation error, and should be "
                "reported to the Embiggen repository as an issue."
            )
    except ModuleNotFoundError as e:
        # If effectively the error is that we cannot load the desired
        # library name, we catch this and re-raise it.
        if f"No module named '{module_library_name}'" == str(e):
            class StubClass(parent_class):

                def __init__(self) :
                    """Raises a useful error message about this class."""
                    super().__init__()
                    self.__class__.__name__ = model_class_name
                    other_candidates = self.find_available_models(
                        self.model_name(),
                        self.task_name()
                    )
                    if other_candidates:
                        other_libraries = [
                            model_name.library_name()
                            for model_name in other_candidates
                        ]
                        other_candidates_message = (
                            "Do be advised that, while this model is not "
                            "currently available on your system "
                            "in this specific library implementation, "
                            f"the same model is implemented in {format_list(other_libraries)}. "
                            "Do be aware that different implementations may have "
                            "very different parametrizations and performance."
                        )
                    else:
                        other_candidates_message = (
                            "At this time, there is no other implementation "
                            f"of the {self.model_name()} model available for "
                            "your system."
                        )
                    raise ModuleNotFoundError(
                        (
                            f"The module {module_library_name} is not available "
                            "on your system "
                            "and therefore we cannot make available the requested "
                            f"model {self.model_name()}, as it is based on the "
                            f"{self.library_name()} library. "
                            "Please do refer to the requested library documentation "
                            f"to proceed with the installation. {other_candidates_message}"
                        )
                    )

                def library_name(self) -> str:
                    """Returns library name of the model."""
                    return formatted_library_name

                def model_name(self) -> str:
                    """Returns model name of the model."""
                    return formatted_model_name

                @staticmethod
                def is_available() -> bool:
                    """Returns whether the model class is actually available in the user system."""
                    return True

            model_class = StubClass
        else:
            # We re-raise the exception.
            raise e

    # We assign the local class as exposed in the
    # caller frame.
    frame.f_back.f_locals[model_class_name] = model_class

    # We register the newly loaded class.
    model_class.register(model_class)
