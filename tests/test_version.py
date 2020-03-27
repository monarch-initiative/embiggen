from validate_version_code import validate_version_code
from xn2v.__version__ import __version__
from unittest import TestCase


class TestTextEncoderSentences(TestCase):

    def test_version(self):
        assert validate_version_code(__version__)
