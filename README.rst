Extended Node2Vec
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability|

Implementation of node2vec with several word2vec family algorithms

The paper node2vec: Scalable Feature Learning for Networks
introduced a method for mapping the nodes of a graph to a low-dimensional
space of features that maximizes the likelihood of preserving
network neighborhoods of nodes.
An approach to generating random walks that resemble either
BFS or DFS traversals was presented.
The random walks generate "sentences" that together can
be regarded as a corpus of text that can be used to generate
embeddings with the word2vec algorithm.

This package allows users to perform node2vec analysis
using several different word2vec family algorithms.

Setting up xn2v
-----------------------------------
To do. We will submit N2V to PyPI as soon as the code is a bit more mature.

Unit testing
-----------------------------------
To run the unit testing on the package, generating
the coverage and the html report, you can use:

.. code:: bash

    nosetests --with-coverage --cover-package=xn2v --cover-html

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get
slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|


.. |travis| image:: https://travis-ci.org/monarch-initiative/n2v.png
   :target: https://travis-ci.org/monarch-initiative/n2v
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=monarch-initiative_n2v&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/monarch-initiative_n2v
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=monarch-initiative_n2v&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/monarch-initiative_n2v
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=monarch-initiative_n2v&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/monarch-initiative_n2v
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/monarch-initiative/N2V/badge.svg?branch=master
    :target: https://coveralls.io/github/monarch-initiative/N2V?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/xn2v.svg
    :target: https://badge.fury.io/py/xn2v
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/xn2v
    :target: https://pepy.tech/badge/xn2v
    :alt: Pypi total project downloads

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/17ecd62a13ee424b87b3fd0b644fdaac
    :target: https://www.codacy.com/gh/monarch-initiative/N2V?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=monarch-initiative/N2V&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/25771b0f4426c0aa425f/maintainability
    :target: https://codeclimate.com/github/monarch-initiative/N2V
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/25771b0f4426c0aa425f/test_coverage
    :target: https://codeclimate.com/github/monarch-initiative/n2v/test_coverage
    :alt: Code Climate Coverate
