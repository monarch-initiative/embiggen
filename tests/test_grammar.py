"""Unit tests to identify quickly potentially grammatical errors in the code base."""

import os
from glob import glob

import pytest


def test_typos():
    """Test for typos in the code base."""
    paths = glob("./embigfgn/**/*.py", recursive=True)

    # List of common typos in the code base relative
    # to the correct spelling of graph-related terms
    # and other such oddities.
    # The dictionary is structured with the correct
    # spelling as the key and a list of common typos
    # as the value.
    typos = {"edge": ["edg", "egde"], "prediction": ["prediction prediction"], "concatenation": ["conatenation"]}

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.lower()
                for correct, incorrect in typos.items():
                    for typo in incorrect:
                        if typo in line:
                            raise ValueError(
                                f"Found typo '{typo}' in line '{line}' in file '{path}'."
                                f" Please correct this typo to '{correct}'."
                            )
