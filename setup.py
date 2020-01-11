from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='n2v',
      version='0.1',
      description='Heterogeneous network node2vec',
      long_description=readme(),
      url='http://github.com/n2v',
      keywords='node2vec HPO',
      author='Vida Ravanmehr, Peter Robinson',
      author_email='vida.ravanmehr@jax.org, peter.robinson@jax.org',
      license='BSD3',
      packages=['n2v'],
      install_requires=[
            'networkx',
            'numpy>=1.16.4',
            'matplotlib',
            'scipy>=1.2.2',
            'pandas',
            'scikit-learn',
            'tensorflow>=2.0',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
