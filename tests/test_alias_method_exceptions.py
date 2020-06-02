import pytest
import numpy as np
from embiggen.graph.alias_method import alias_setup

def test_alias_setup_exceptions():
    with pytest.raises(ValueError):
        alias_setup(np.array([]))
    
    with pytest.raises(ValueError):
        alias_setup(np.array([0]))