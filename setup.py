"""Module installing the Embiggen package."""
from setuptools import find_packages, setup
from codecs import open as copen
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def readme():
    with open('README.rst') as f:
        return f.read()


test_deps = [
    "codacy-coverage",
    "coveralls",
    'pytest',
    "pytest-cov",
    "validate_version_code",
    "pylint",
    "silence_tensorflow"
]

extras = {
    'test': test_deps,
    'nltk': [
        "pytest",
        "nltk"
    ]
}

def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("embiggen", "__version__.py")

# TODO: Authors add your emails!!!
authors = {
    "Vida Ravanmehr": "vida.ravanmehr@jax.org",
    "Peter Robinson": "peter.robinson@jax.org",
    "Luca Cappelletti": "luca.cappelletti1@unimi.it",
    "Tommaso Fontana": "tommaso.fontana@mail.polimi.it"
}

setup(
    name='embiggen',
    version=__version__,
    description='Graph embedding, machine learning, and visualization library.',
    long_description=readme(),
    url='https://github.com/monarch-initiative/embiggen',
    keywords='node2vec,word2vec,CBOW,SkipGram,GloVe',
    author=", ".join(list(authors.keys())),
    author_email=", ".join(list(authors.values())),
    license='BSD3',
    python_requires='>=3.6.0',
    packages=find_packages(
        exclude=['contrib', 'docs', 'tests*', 'notebooks*']),
    install_requires=[
        'numpy',
        'pandas',
        "tqdm",
        "matplotlib>=3.5.2",
        "scikit-learn",
        "ddd_subplots>=1.0.19",
        "sanitize_ml_labels>=1.0.38",
        "keras_mixed_sequence>=1.0.28",
        "extra_keras_metrics>=2.0.7",
        "ensmallen>=0.7.0.dev21",
        "validate_version_code",
        "cache_decorator>=2.1.6",
        "packaging"
    ],
    tests_require=test_deps,
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
