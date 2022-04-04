from setuptools import find_packages, setup


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

# TODO: Authors add your emails!!!
authors = {
    "Vida Ravanmehr": "vida.ravanmehr@jax.org",
    "Peter Robinson": "peter.robinson@jax.org",
    "Luca Cappelletti": "luca.cappelletti1@unimi.it",
    "Tommaso Fontana": "tommaso.fontana@mail.polimi.it"
}

setup(
    name='embiggen',
    version='0.10.0.dev5',
    description='Graph machine learning and visualization library.',
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
        "matplotlib",
        "sklearn",
        "ddd_subplots>=1.0.19",
        "sanitize_ml_labels>=1.0.29",
        "keras_mixed_sequence>=1.0.28",
        "extra_keras_metrics>=2.0.5",
        "ensmallen>=0.7.0.dev8",
        "validate_version_code",
        "packaging"
    ],
    tests_require=test_deps,
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
