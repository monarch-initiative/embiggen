"""Test to validate that version code is written in PEP8 compliant format."""
from validate_version_code import validate_version_code
from embiggen.__version__ import __version__


def test_version():
    """Test to validate that version code is written in PEP8 compliant format."""
    assert validate_version_code(__version__)
