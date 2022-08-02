import pytest
from embiggen.utils import number_to_ordinal
from unittest import TestCase


class TestOrdinal(TestCase):

    def setUp(self):
        pass

    def test_number_to_ordinal(self):
        self.assertEqual(number_to_ordinal(5), "Fifth")
        self.assertEqual(number_to_ordinal(30), "Thirtieth")
        self.assertEqual(number_to_ordinal(21), "TwentyFirst")

        with pytest.raises(NotImplementedError):
            number_to_ordinal(-60)

        with pytest.raises(NotImplementedError):
            number_to_ordinal(600)
