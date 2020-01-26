from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


test_deps = [
    "codacy-coverage",
    "coveralls",
    'nose',
    "nose-cov",
    "validate_version_code"
]

extras = {
    'test': test_deps,
}

setup(
    name='n2v',
    version='0.2',
    description='Implementation of node2vec with several word2vec family algorithms',
    long_description=readme(),
    url='https://github.com/TheJacksonLaboratory/N2V',
    keywords='node2vec. word2vec',
    author='Vida Ravanmehr, Peter Robinson',
    author_email='vida.ravanmehr@jax.org, peter.robinson@jax.org',
    license='BSD3',
    packages=['n2v'],
    install_requires=[
        'silence_tensorflow',
        'numpy>=1.16.4',
        'pandas',
        'sklearn',
        'tensorflow==2.0',
    ],
    test_suite='nose.collector',
    tests_require=test_deps,
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
)
