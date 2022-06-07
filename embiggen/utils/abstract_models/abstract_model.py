"""Module providing generic abstract model."""
from embiggen.utils.abstract_models.list_formatting import format_list
from typing import Dict, Any, Type, List, Optional
from dict_hash import Hashable, sha256
from userinput.utils import closest


def abstract_class(klass: Type["AbstractModel"]) -> Type["AbstractModel"]:
    """Simply adds a descriptor for meta-programming and nothing else."""
    return klass

@abstract_class
class AbstractModel(Hashable):
    """Class defining properties of a generic abstract model."""

    MODELS_LIBRARY: Dict[str, Dict[str, Dict[str, Type["AbstractModel"]]]] = {}

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Return parameters to create a model with minimal configuration to test execution."""
        raise NotImplementedError((
            "The `smoke_test_parameters` method must be implemented "
            "in the child classes of abstract model."
        ))
    
    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        raise NotImplementedError((
            "The `parameters` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def requires_edge_weights() -> bool:
        """Returns whether the model requires edge weights."""
        raise NotImplementedError((
            "The `requires_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def task_involves_edge_weights() -> bool:
        """Returns whether the model task involves edge weights."""
        raise NotImplementedError((
            "The `task_involves_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        raise NotImplementedError((
            "The `can_use_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        raise NotImplementedError((
            "The `is_using_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        """Returns whether the model requires positive edge weights."""
        raise NotImplementedError((
            "The `requires_positive_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def task_involves_topology() -> bool:
        """Returns whether the model task involves topology."""
        raise NotImplementedError((
            "The `task_involves_topology` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def is_topological() -> bool:
        """Returns whether this embedding is based on graph topology."""
        raise NotImplementedError((
            "The `is_topological` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def requires_node_types() -> bool:
        """Returns whether the model requires node types."""
        raise NotImplementedError((
            "The `requires_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def task_involves_node_types() -> bool:
        """Returns whether the model task involves node types."""
        raise NotImplementedError((
            "The `task_involves_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        raise NotImplementedError((
            "The `can_use_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        raise NotImplementedError((
            "The `can_use_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def requires_edge_types() -> bool:
        """Returns whether the model requires edge types."""
        raise NotImplementedError((
            "The `requires_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def task_involves_edge_types() -> bool:
        """Returns whether the model task involves edge types."""
        raise NotImplementedError((
            "The `task_involves_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        raise NotImplementedError((
            "The `can_use_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        raise NotImplementedError((
            "The `can_use_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))
    
    @staticmethod
    def task_name() -> str:
        """Returns the task for which this model is being used."""
        raise NotImplementedError((
            "The `task_name` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def library_name() -> str:
        """Returns library name of the model."""
        raise NotImplementedError((
            "The `library_name` method must be implemented "
            "in the child classes of abstract model."
        ))

    @staticmethod
    def model_name() -> str:
        """Returns model name of the model."""
        raise NotImplementedError((
            "The `model_name` method must be implemented "
            "in the child classes of abstract model."
        ))

    def clone(self) -> Type["AbstractModel"]:
        """Returns copy of the current model."""
        raise NotImplementedError((
            "The `clone` method must be implemented in "
            "the child classes of the abstract model."
        ))

    def consistent_hash(self) -> str:
        """Returns consistent hash describing the model."""
        return sha256({
            **self.parameters(),
            "model_name": self.model_name(),
            "library_name": self.library_name(),
            "task_name": self.task_name(),
        })
    
    @staticmethod
    def is_available() -> bool:
        """Returns whether the model class is actually available in the user system."""
        return True

    @staticmethod
    def get_model_data(
        model_name: str
    ) -> Dict[str, Dict]:
        """Returns data relative to the registered model data."""
        # We check if the provided string is not an empty string.
        if len(model_name) == 0:
            raise ValueError(
                "The provided model name is empty."
            )

        # We turn this to lowercase in order to allow
        # for error in casing of the model, since one may
        # write models like `GloVe` also as `Glove` or other
        # typos, which are generally easy to make.
        lowercase_model_mapping = AbstractModel.get_available_model_names_in_lowercase_mapping()
        if model_name.lower() not in lowercase_model_mapping:
            raise ValueError(
                f"The provided model name `{model_name}` is not available. "
                f"Did you mean {closest(model_name, lowercase_model_mapping.values())}?"
            )
        # We retrieve the model standard name.
        model_name = lowercase_model_mapping[model_name.lower()]
        return AbstractModel.MODELS_LIBRARY[model_name]

    @staticmethod
    def get_task_data(
        model_name: str,
        task_name: str
    ) -> Dict[str, Dict]:
        """Returns data relative to the registered model and task data."""
        model_data = AbstractModel.get_model_data(model_name)

        # We check if the provided string is not an empty string.
        if len(task_name) == 0:
            raise ValueError(
                "The provided task name is empty."
            )

        # We do a similar check as the one above for the tasks,
        # as one may do typos while writig the task name and
        # we should always provide the best possible help message.
        lowercase_task_mapping = {
            t.lower(): t
            for t in model_data.keys()
        }
        if task_name.lower() not in lowercase_task_mapping:
            raise ValueError(
                f"The provided task name `{task_name}` is not available for "
                f"the requested model {model_name}."
                f"Did you mean {closest(task_name, lowercase_task_mapping.values())}?"
            )

        # We retrieve the task standard name.
        task_name = lowercase_task_mapping[task_name.lower()]

        # We retrieve the task data.
        return model_data[task_name]

    @staticmethod
    def get_library_data(
        model_name: str,
        task_name: str,
        library_name: str
    ) -> Type["AbstractModel"]:
        """Returns model relative library, task and model name."""
        task_data = AbstractModel.get_task_data(model_name, task_name)

        # We check if the provided string is not an empty string.
        if len(library_name) == 0:
            raise ValueError(
                "The provided library name is empty."
            )

        lowercase_libraries_mapping = {
            t.lower(): t
            for t in task_data.keys()
        }
        if library_name.lower() not in lowercase_libraries_mapping:
            raise ValueError(
                f"The provided library name `{library_name}` is not available for "
                f"the requested model {model_name}. "
                f"Did you mean {closest(library_name, lowercase_libraries_mapping.values())}?"
            )

        # We retrieve the library standard name.
        library_name = lowercase_libraries_mapping[library_name.lower()]

        # We retrieve the library data.
        return task_data[library_name]

    @staticmethod
    def get_model_from_library(
        model_name: str,
        task_name: Optional[str] = None,
        library_name: Optional[str] = None,
    ) -> Type["AbstractModel"]:
        """Returns list of models implementations available for given task and model.

        Parameters
        -------------------
        model_name: str
            The name of the model to retrieve.
        task_name: Optional[str] = None
            The task that this implementation of the model should be able to do.
            If not provided, it will be returned the model if it has only a single
            possible task. If multiple tasks are available, an exception will
            be raised.
        library_name: Optional[str] = None
            The library from which to get the implementation of this model.
            If not provided, it will be returned the model if it has only a single
            possible library. If multiple librarys are available, an exception will
            be raised.
        """
        if task_name is None:
            task_names = list(AbstractModel.get_model_data(model_name).keys())
            if len(task_names) == 1:
                task_name = task_names[0]
            else:
                formatted_list = format_list(task_names)
                raise ValueError(
                    f"The requested model `{model_name}` is available for "
                    "multiple tasks and no specific task was requested, "
                    "so it is unclear which task you intend to execute. "
                    f"Specifically, the available tasks are {formatted_list}."
                    "Please do provide a task name to resolve this ambiguity."
                )

        task_data = AbstractModel.get_task_data(model_name, task_name)

        if library_name is None:
            library_names = list(task_data.keys())
            if len(library_names) == 1:
                library_name = library_names[0]
            elif "Ensmallen" in library_names:
                library_name = "Ensmallen"
            else:
                formatted_list = format_list(library_names)
                raise ValueError(
                    (
                        f"The requested model `{model_name}` is available for "
                        "multiple libraries and no specific library was requested, "
                        "so it is unclear which library you intend to execute. "
                        f"Specifically, the available libraries are {formatted_list}. "
                        "Please do provide a library name to resolve this ambiguity."
                    )
                )

        model_class = AbstractModel.get_library_data(
            model_name,
            task_name,
            library_name
        )

        # If the model is not available, we just
        # instantiate it to cause its helpful ModuleNotFound
        # exception raise, with the informations to help the user.
        if not model_class.is_available():
            model_class()

        # Otherwise if the model is available, we return
        # its class to let the user do whathever they want.
        return model_class

    @staticmethod
    def get_available_model_names() -> List[str]:
        """Returns list of available model names."""
        return list(AbstractModel.MODELS_LIBRARY.keys())

    @staticmethod
    def get_available_model_names_in_lowercase_mapping() -> Dict[str, str]:
        """Returns list of available model names in lowercase."""
        return {
            model_name.lower(): model_name
            for model_name in AbstractModel.get_available_model_names()
        }

    @staticmethod
    def find_available_models(
        model_name: str,
        task_name: str
    ) -> List[Type["AbstractModel"]]:
        """Returns list of models implementations available for given task and model.

        Parameters
        -------------------
        model_name: str
            The name of the model to retrieve.
        task_name: str
            The task that this implementation of the model should be able to do.
        """
        return [
            model
            for model in AbstractModel.get_task_data(model_name, task_name).values()
            if model.is_available()
        ]

    @staticmethod
    def register(model_class: Type["AbstractModel"]):
        """Registers the provided model in the model library.

        Parameters
        ------------------
        model_class:  Type["AbstractModel"]
            The class to register.
        """
        model_name = model_class.model_name()
        # If this is the first model of its kind to be registered.
        if model_name not in AbstractModel.MODELS_LIBRARY:
            AbstractModel.MODELS_LIBRARY[model_name] = {}

        # We retrieve the data for the model to enrich it.
        # This is NOT a copy, but a reference to the same STATIC object.
        model_data = AbstractModel.MODELS_LIBRARY[model_name]

        task_name = model_class.task_name()
        if task_name not in model_data:
            model_data[task_name] = {}

        task_data = model_data[task_name]

        class_name = model_class.__name__

        library_name = model_class.library_name()
        if library_name not in task_data:
            task_data[library_name] = model_class
        else:
            raise ValueError(
                f"The provided model called `{model_name}` with class name "
                f"`{class_name}`, implemented using the {library_name} library "
                "was already previously registered as available for the "
                f"`{task_name}` task. This is an implementation issue, "
                "so if you are seeing this problem either you are trying "
                "to register a custom model or you have found an error in "
                "the Embiggen library. If you believe this to be the latter "
                "please do open an issue in the Embiggen repository."
            )
