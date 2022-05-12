"""Submodule to automatically create __init__ files for the library submodules with stubs."""
from typing import Dict, List, Type
from .abstract_model import AbstractModel
import ast
from ast import ClassDef, ImportFrom, FunctionDef, Return
from glob import glob
from .model_stub import get_model_or_stub
import os
import inspect
import traceback


def get_python_code_from_import(
    original_file_path: str,
    import_from: Dict
):
    element = import_from["element"]
    source_path = os.path.join(
        *original_file_path.split(os.sep)[:-element.level],
        f"{element.module}.py"
    )

    if not os.path.exists(source_path):
        source_path = os.path.join(
            *original_file_path.split(os.sep)[:-element.level],
            f"{element.module}/__init__.py"
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
            parent_names.extend(get_class_parent_names(
                *get_python_code_from_import(
                    original_file_path,
                    import_from
                ),
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
            "name": alias.name
        }
        for element in parsed.body
        if isinstance(element, ImportFrom)
        for alias in element.names
        if element.level > 0
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
        if not isinstance(function, FunctionDef) and function.name == "model_name":
            continue
        for function_line in function.body:
            if isinstance(function_line, Return):
                return function_line.value.s
    raise ValueError(
        "Unable to find the method `model_name` in the "
        f"model class {klass.name}."
    )


def build_init(
    module_library_name: str,
    formatted_library_name: str,
    expected_parent_class: Type[AbstractModel]
):
    """Create the init for this submodule.

    Parameters
    --------------------
    module_library_name: str
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
            if len(klass.decorator_list) > 0 and klass.decorator_list[0].id == "abstract_class":
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
                    module_library_name=module_library_name,
                    formatted_library_name=formatted_library_name,
                    submodule_name=submodule_name,
                    model_class_name=klass.name,
                    formatted_model_name=find_method_name(klass),
                    parent_class=expected_parent_class
                )

    frame.f_back.f_locals["__all__"] = generated_classes
    
    del frame
