from validate_version_code import validate_version_code
from embiggen.__version__ import __version__
from unittest import TestCase


class TestTextEncoderSentences(TestCase):

    def test_version(self):
        self.assertTrue(validate_version_code(__version__))
