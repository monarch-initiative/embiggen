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
    'visualization': [
        "cmake",
        "MulticoreTSNE",
        "matplotlib",
        "sanitize_ml_labels>=1.0.12"
    ]
}

# TODO: Authors add your emails!!!
authors = {
    "Vida Ravanmehr":"vida.ravanmehr@jax.org",
    "Peter Robinson":"peter.robinson@jax.org",
    "Luca Cappelletti":"luca.cappelletti1@unimi.it",
    "Tommaso Fontana":"tommaso.fontana@mail.polimi.it"
}

setup(
    name='embiggen',
    version='0.6.0',
    description='Extended implementation of node2vec with several word2vec family algorithms',
    long_description=readme(),
    url='https://github.com/monarch-initiative/embiggen',
    keywords='node2vec,word2vec,CBOW,SkipGram,GloVe',
    author=", ".join(list(authors.keys())),
    author_email=", ".join(list(authors.values())),
    license='BSD3',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'notebooks*']),
    install_requires=[
        'numpy',
        'pandas',
        "nltk",
        'tensorflow>=2.0.0',
        "ensmallen_graph>=0.3.1",
        "keras_mixed_sequence>=1.0.13"
    ],
    tests_require=test_deps,
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
