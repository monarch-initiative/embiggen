"""Module providing generic abstract model."""
from typing import Callable
from embiggen.utils.abstract_models.list_formatting import format_list
from typing import Dict, Any, Type, List, Optional
from dict_hash import Hashable, sha256
from userinput.utils import must_be_in_set
import inspect


def abstract_class(klass: Type["AbstractModel"]) -> Type["AbstractModel"]:
    """Simply adds a descriptor for meta-programming and nothing else."""
    return klass

def is_not_implemented(method: Callable) -> bool:
    """Returns whether this method contains a raise for not being implemented."""
    return "raise NotImplementedError" in inspect.getsource(method)


def is_implemented(method: Callable) -> bool:
    """Returns whether this method is implemented."""
    return not is_not_implemented(method)


@abstract_class
class AbstractModel(Hashable):
    """Class defining properties of a generic abstract model."""

    MODELS_LIBRARY: Dict[str, Dict[str, Dict[str, Type["AbstractModel"]]]] = {}

    def __init__(self, random_state: Optional[int] = None):
        """Create new abstract model.

        Parameters
        ---------------
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        super().__init__()
        if self.is_stocastic() and random_state is None:
            raise ValueError(
                "The provided model is stocastic, but no "
                "random state was provided."
            )
        if not self.is_stocastic() and random_state is not None:
            raise ValueError(
                "The provided model is not stocastic, yet a "
                f"random state of `{random_state}` was provided."
            )

        can_use_edge_weights = self.__getattribute__("can_use_edge_weights")
        requires_positive_edge_weights = self.__getattribute__(
            "requires_positive_edge_weights")

        if (
            is_implemented(can_use_edge_weights) and
            not can_use_edge_weights() and
            is_implemented(requires_positive_edge_weights)
        ):
            raise ValueError(
                "We have found an useless method in the "
                f"class {self.__class__.__name__}, implementing method "
                f"{self.model_name()} from library {self.library_name()} "
                f"and task {self.task_name()}. "
                "It does not make sense to implement the "
                f"`requires_positive_edge_weights` method when the `can_use_edge_weights` "
                "always returns False, as it is already handled "
                "in the root abstract model class."
            )

        # Identify and resolve tautological implementations.
        for graph_property in ("edge_types", "node_types", "edge_weights"):
            requires = f"requires_{graph_property}"
            requires_method = self.__getattribute__(requires)
            can_use = f"can_use_{graph_property}"
            can_use_method = self.__getattribute__(can_use)
            is_using = f"is_using_{graph_property}"

            if (
                is_not_implemented(requires_method) and
                is_not_implemented(can_use_method)
            ):
                raise ValueError(
                    "We have found a missing method implementation in the "
                    f"class {self.__class__.__name__}, implementing method "
                    f"{self.model_name()} from library {self.library_name()} "
                    f"and task {self.task_name()}. "
                    f"It is strictly necessary to implement either the `{requires}` "
                    f"method or the {can_use} method in order to adhere to the model "
                    "interface and facilitate the integration with the pipelines."
                )

            if is_implemented(requires_method) and requires_method():
                for method in (can_use, is_using):
                    if is_implemented(self.__getattribute__(method)):
                        raise ValueError(
                            "We have found an useless method in the "
                            f"class {self.__class__.__name__}, implementing method "
                            f"{self.model_name()} from library {self.library_name()} "
                            f"and task {self.task_name()}. "
                            "It does not make sense to implement the "
                            f"`{method}` method when the `{requires}` "
                            "always returns True, as it is already handled "
                            "in the root abstract model class."
                        )
            if is_implemented(can_use_method) and not can_use_method():
                for method in (requires, is_using):
                    if is_implemented(self.__getattribute__(method)):
                        raise ValueError(
                            "We have found an useless method in the "
                            f"class {self.__class__.__name__}, implementing method "
                            f"{self.model_name()} from library {self.library_name()} "
                            f"and task {self.task_name()}. "
                            "It does not make sense to implement the "
                            f"`{method}` method when the `{can_use}` "
                            "always returns False, as it is already handled "
                            "in the root abstract model class."
                        )

        self._random_state = random_state

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Return parameters to create a model with minimal configuration to test execution."""
        raise NotImplementedError((
            "The `smoke_test_parameters` method must be implemented "
            "in the child classes of abstract model."
        ))

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        if self._random_state is None:
            return dict()
        return dict(
            random_state=self._random_state
        )

    def into_smoke_test(self) -> "Self":
        """Creates new instance with smoke test parameters."""
        return self.__class__(**{
            **self.parameters(),
            **self.smoke_test_parameters()
        })

    @classmethod
    def requires_edge_weights(cls) -> bool:
        """Returns whether the model requires edge weights."""
        try:
            if not cls.can_use_edge_weights():
                return False
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `requires_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def task_involves_edge_weights(cls) -> bool:
        """Returns whether the model task involves edge weights."""
        raise NotImplementedError((
            "The `task_involves_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        try:
            if (
                is_implemented(cls.requires_edge_weights) and
                cls.requires_edge_weights()
            ):
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `can_use_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        try:
            if self.requires_edge_weights():
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `is_using_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def requires_positive_edge_weights(cls) -> bool:
        """Returns whether the model requires positive edge weights."""
        try:
            if not cls.requires_edge_weights():
                return False
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `requires_positive_edge_weights` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def task_involves_topology(cls) -> bool:
        """Returns whether the model task involves topology."""
        raise NotImplementedError((
            "The `task_involves_topology` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def is_topological(cls) -> bool:
        """Returns whether this embedding is based on graph topology."""
        raise NotImplementedError((
            "The `is_topological` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def requires_node_types(cls) -> bool:
        """Returns whether the model requires node types."""
        try:
            if not cls.can_use_node_types():
                return False
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `requires_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def task_involves_node_types(cls) -> bool:
        """Returns whether the model task involves node types."""
        raise NotImplementedError((
            "The `task_involves_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        try:
            if cls.requires_node_types():
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `can_use_node_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        try:
            if self.requires_node_types():
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `is_using_node_types` method must be implemented "
            "in the child classes of abstract model, but was not implemented "
            f"in the class {self.__class__.__name__} implementing the model {self.model_name()} "
            f"from the library {self.library_name()}."
        ))

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Returns whether the model requires edge types."""
        try:
            if not cls.can_use_edge_types():
                return False
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `requires_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def task_involves_edge_types(cls) -> bool:
        """Returns whether the model task involves edge types."""
        raise NotImplementedError((
            "The `task_involves_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        try:
            if cls.requires_edge_types():
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `can_use_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        try:
            if self.requires_edge_types():
                return True
        except NotImplementedError:
            pass
        raise NotImplementedError((
            "The `is_using_edge_types` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def task_name(cls) -> str:
        """Returns the task for which this model is being used."""
        raise NotImplementedError((
            "The `task_name` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def library_name(cls) -> str:
        """Returns library name of the model."""
        raise NotImplementedError((
            "The `library_name` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def model_name(cls) -> str:
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
        return sha256(dict(
            **self.parameters(),
            model_name=self.model_name(),
            library_name=self.library_name(),
            task_name=self.task_name(),
        ))

    @staticmethod
    def is_available() -> bool:
        """Returns whether the model class is actually available in the user system."""
        return True

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        raise NotImplementedError((
            "The `is_stocastic` method must be implemented "
            "in the child classes of abstract model."
        ))

    def set_random_state(self, random_state: int):
        """Returns whether the model is stocastic and has therefore a random state."""
        if not self.is_stocastic():
            raise ValueError(
                "It does not make sense to set the random state of a "
                "non-stocastic model."
            )
        self._random_state = random_state

    @staticmethod
    def get_task_data(
        model_name: str,
        task_name: str
    ) -> Dict[str, Dict]:
        """Returns data relative to the registered model and task data."""
        if len(model_name) == 0:
            raise ValueError(
                "The provided model name is empty."
            )

        # We check if the provided string is not an empty string.
        if len(task_name) == 0:
            raise ValueError(
                "The provided task name is empty."
            )

        task_name = must_be_in_set(
            task_name,
            AbstractModel.MODELS_LIBRARY,
            "task name"
        )

        model_name = must_be_in_set(
            model_name,
            AbstractModel.MODELS_LIBRARY[task_name],
            "model name"
        )

        # We retrieve the task data.
        return AbstractModel.MODELS_LIBRARY[task_name][model_name]

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

        library_name = must_be_in_set(
            library_name,
            task_data.keys(),
            "library name"
        )

        # We retrieve the library data.
        return task_data[library_name]

    @classmethod
    def get_model_from_library(
        cls,
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
            task_name = cls.task_name()

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
        task_name = model_class.task_name()
        model_name = model_class.model_name()
        # If this is the first model of its kind to be registered.
        if task_name not in AbstractModel.MODELS_LIBRARY:
            AbstractModel.MODELS_LIBRARY[task_name] = {}

        # We retrieve the data for the model to enrich it.
        # This is NOT a copy, but a reference to the same STATIC object.
        model_data = AbstractModel.MODELS_LIBRARY[task_name]

        if model_name not in model_data:
            model_data[model_name] = {}

        task_data = model_data[model_name]

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
