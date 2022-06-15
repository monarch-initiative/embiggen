"""Submodule to automatically create __init__ files for the library submodules with stubs."""
from typing import Dict, List, Type, Union

from embiggen.utils.abstract_models.abstract_model import AbstractModel
import ast
from ast import ClassDef, ImportFrom, FunctionDef, Return
from glob import glob
from embiggen.utils.abstract_models.model_stub import get_model_or_stub
import os
import inspect
import traceback


def get_python_code_from_import(
    original_file_path: str,
    import_from: Dict
):
    element = import_from["element"]
    file_parts = original_file_path.split(os.sep)

    found = 0
    for i, path_chunk in enumerate(reversed(file_parts)):
        if path_chunk == "embiggen":
            found = i + 1
            break
    
    path = os.sep.join(file_parts[:-found] + element.module.split("."))
    source_path = f"{path}.py"

    if not os.path.exists(source_path):
        source_path = f"{path}/__init__.py"

    # While this secondary check should NEVER be necessary
    # sadly pip has some issues with removing deleted files
    # when updating the packages. In order to avoid
    # such accidents, we double check this case.
    if not os.path.exists(source_path):
        raise ValueError(
            "If you see this error, there may be an issue "
            "in your embiggen installation. For instance, "
            "you may have multiple versions installed at once "
            "which are currently ad odds with one another. "
            "Often, this is caused when an older version had a "
            "file that is no longer present but was not deleted "
            "during the installation process. The file causing this issue are "
            f"'{source_path}' and '{original_file_path}'. Consider deleting it."
        )

    expected_class_name = import_from["name"]

    with open(source_path, "r") as f:
        python_code = f.read()

    parsed = ast.parse(python_code)
    klasses = get_classes(parsed)
    imports = get_imports(parsed)

    desired_klass = None

    # We search for the class here
    for klass in klasses:
        if klass.name == expected_class_name:
            desired_klass = klass

    # We need to go look for it in the imports
    for import_name, import_from in imports.items():
        # If this is a parent class
        if import_name == expected_class_name:
            # Get the imported path and parsed code.
            return get_python_code_from_import(
                source_path,
                import_from
            )

    return (
        original_file_path,
        desired_klass,
        klasses,
        imports
    )


def get_class_parent_names(
    original_file_path: str,
    klass: ClassDef,
    klasses: List[ClassDef],
    imports: List[Dict],
    expected_parent: str
) -> List[str]:
    """Return list of parent classes names."""

    if klass is None:
        return []

    # First we search for the parent class in the same file.
    parent_names = [
        base.id
        for base in klass.bases
    ]

    if expected_parent in parent_names:
        return parent_names

    # If the class does not have parents
    if len(parent_names) == 0:
        return []

    # Otherwise we search for the parents in this same directory
    for candidate_parent in klasses:
        # If this is a parent class
        if candidate_parent.name in parent_names:
            # We find the parents of this class
            parent_names.extend(
                get_class_parent_names(
                    original_file_path,
                    candidate_parent,
                    klasses
                )
            )

            if expected_parent in parent_names:
                return parent_names

    # Then we check if any of the imports are parents
    for import_name, import_from in imports.items():
        # If this is a parent class
        if import_name in parent_names:
            # Get the imported path and parsed code.
            results = get_python_code_from_import(
                original_file_path,
                import_from
            )
            if results:
                parent_names.extend(get_class_parent_names(
                    *results,
                    expected_parent
                ))

            if expected_parent in parent_names:
                return parent_names

    return parent_names


def get_imports(parsed) -> Dict[str, ImportFrom]:
    """Returns local imports identified in parsed Python code."""
    return {
        getattr(alias, "as_name", alias.name): {
            "element": element,
            "name": alias.name,
        }
        for element in parsed.body
        if isinstance(element, ImportFrom)
        for alias in element.names
        if element.module.startswith("embiggen")
    }


def get_classes(parsed) -> List[ClassDef]:
    """Returns classes identified in parsed Python code."""
    return [
        element
        for element in parsed.body
        if isinstance(element, ClassDef)
    ]


def find_method_name(klass: ClassDef) -> str:
    """Returns name extracted from class."""
    for function in klass.body:
        if not isinstance(function, FunctionDef) or function.name != "model_name":
            continue
        for function_line in function.body:
            if isinstance(function_line, Return):
                return function_line.value.s
    raise ValueError(
        "Unable to find the method `model_name` in the "
        f"model class {klass.name}."
    )


def build_init(
    module_library_names: Union[str, List[str]],
    formatted_library_name: str,
    expected_parent_class: Type[AbstractModel]
):
    """Create the init for this submodule.

    Parameters
    --------------------
    module_library_names: Union[str, List[str]]
        Name of the library dependency to be check for availability.
    formatted_library_name: str
        The formatted name of the library for visualization pourposes.
    expected_parent_class: Type[AbstractModel]
        The class to check for.
    """
    path_pattern = os.path.join(os.path.dirname(os.path.abspath(
        traceback.extract_stack()[-2].filename
    )), "*.py")

    # We retrieve the context of the caller.
    frame = inspect.currentframe()

    generated_classes = []

    for path in glob(path_pattern):
        submodule_name = path.split(os.sep)[-1].split(".")[0]
        with open(path, "r") as f:
            python_code = f.read()
        parsed = ast.parse(python_code)
        klasses = get_classes(parsed)
        imports = get_imports(parsed)
        for klass in klasses:
            # If this class is an abstract.
            if (
                len(klass.decorator_list) > 0 and
                klass.decorator_list[0].id == "abstract_class" or
                klass.name == expected_parent_class.__name__
            ):
                continue
            # If this class has the expected parent.
            if expected_parent_class.__name__ in get_class_parent_names(
                path,
                klass,
                klasses,
                imports,
                expected_parent_class.__name__
            ):
                generated_classes.append(klass.name)
                get_model_or_stub(
                    frame,
                    module_library_names=module_library_names,
                    formatted_library_name=formatted_library_name,
                    submodule_name=submodule_name,
                    model_class_name=klass.name,
                    formatted_model_name=find_method_name(klass),
                    parent_class=expected_parent_class
                )

    frame.f_back.f_locals["__all__"] = generated_classes

    del frame
