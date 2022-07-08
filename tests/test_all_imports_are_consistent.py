import pytest
import inspect
import embiggen

def check_alls(module):
    alls = getattr(module, "__all__", None)
    if alls is None:
        return

    for sub in alls:
        assert sub in dir(module), "{} is in the __all__ of module {} but cannot be imported".format(sub, module.__name__)

    for value in vars(module).values():
        # only check modules inside embiggen
        mod = inspect.getmodule(value)
        if mod is None or "embiggen" not in mod.__name__:
            continue

        if inspect.ismodule(value):
            check_alls(value)

def test_alls():
    check_alls(embiggen)
