import pytest
from embiggen.utils.sklearn_utils import (
    is_sklearn_classifier_model,
    must_be_an_sklearn_classifier_model
)
from sklearn.tree import DecisionTreeClassifier
from unittest import TestCase


class TestSklearnUtils(TestCase):

    def setUp(self):
        pass

    def test_sklearn_utils(self):
        self.assertTrue(
            is_sklearn_classifier_model(DecisionTreeClassifier())
        )
        self.assertFalse(
            is_sklearn_classifier_model(DecisionTreeClassifier)
        )
        self.assertFalse(
            is_sklearn_classifier_model(int)
        )
        self.assertFalse(
            is_sklearn_classifier_model(6)
        )
        must_be_an_sklearn_classifier_model(
            DecisionTreeClassifier()
        )
        with pytest.raises(ValueError):
            must_be_an_sklearn_classifier_model(
                DecisionTreeClassifier
            )
        with pytest.raises(ValueError):
            must_be_an_sklearn_classifier_model(
                67
            )
