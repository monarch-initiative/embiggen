from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


test_deps = [
    "codacy-coverage",
    "coveralls",
    'nose',
    "nose-cov",
    "validate_version_code",
    "pylint",
    "silence_tensorflow",
    "parameterized"
]

extras = {
    'test': test_deps,
}

setup(
    name='embiggen',
    version='0.6.0',
    description='Extended implementation of node2vec with several word2vec family algorithms',
    long_description=readme(),
    url='https://github.com/monarch-initiative/N2V',
    keywords='node2vec. word2vec',
    author='Vida Ravanmehr, Peter Robinson',
    author_email='vida.ravanmehr@jax.org, peter.robinson@jax.org',
    license='BSD3',
    packages=['embiggen'],
    install_requires=[
        'click',
        'keras',
        'setuptools>=42.0.0',
        'numpy>=1.16.4',
        'pandas',
        'parameterized',
        'silence_tensorflow',
        'sklearn',
        'nltk',
        'more_itertools',
        'tqdm',
        "environments_utils",
        "keras_tqdm",
        'tensorflow>=2.0.0',
        'click',
        'sanitize_ml_labels',
        'deflate_dict',
        'cache_decorator',
        "numba"
    ],
    test_suite='nose.collector',
    tests_require=test_deps,
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
