import pytest
import inspect
import embiggen
import warnings

def check_alls(module):
    error_msg = ""

    # Check for module documentation of __init__ files
    if inspect.getfile(module).endswith("__init__.py"):
        doc = getattr(module, "__doc__", None)
        if doc is None or module.__doc__.strip() == "":
            warnings.warn("The file '{}' doesn't have documentation!".format(inspect.getfile(module)))

    alls = getattr(module, "__all__", None)
    # warn about the missing __all__ and return
    if alls is None:
        if inspect.getfile(module).endswith("__init__.py"):
            warnings.warn("The module '{}' doesn't have an __all__".format(module.__name__))
    else:
        # Check that all the values in all resolves
        missings = set()
        for sub in alls:
            if sub not in dir(module):
                missings.add(sub)

        if len(missings) > 0:
            error_msg += "in the __all__ of module '{}' the following values '{}' cannot be imported\n".format(
                module.__name__, missings,
            )

        # Check for duplicates
        if len(alls) != len(set(alls)):
            duplicates = list(sorted({
                x
                for x in alls
                if alls.count(x) > 1
            }))
            error_msg += "is in the __all__ of module '{}' the following keys are duplicated '{}'\n".format(
                module.__name__, duplicates,
            )

        if len(error_msg) > 0:
            raise AssertionError(error_msg)

    # Recursive step for submodules (even if not exported in __all__)
    for value in vars(module).values():
        # only check modules inside embiggen
        mod = inspect.getmodule(value)
        if mod is None or "embiggen" not in mod.__name__:
            continue

        if inspect.ismodule(value):
            check_alls(value)

def test_alls():
    check_alls(embiggen)
