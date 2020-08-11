"""Test to validate that version code is written in PEP8 compliant format."""
from unittest import TestCase
from validate_version_code import validate_version_code
from embiggen.__version__ import __version__


class TestTextTransformerSentences(TestCase):
    """Unit test to validate that version code is compliant to PEP8 format."""

    def test_version(self):
        """Test to validate version code."""
        self.assertTrue(validate_version_code(__version__))
