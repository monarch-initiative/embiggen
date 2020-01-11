####################################################
N2V: A python library for node2vec family algorithms
####################################################

The paper `node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>`_ introduced
a method for mapping the nodes of a graph to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes.
An approach to generating random walks that resemble either BFS or DFS traversals was presented. The
random walks generate "sentences" that together can be regarded as a corpus of text
that can be used to generate embeddings with the `word2vec <https://arxiv.org/abs/1301.3781>`_
algorithm.

This package allows users to perform node2vec analysis using several different
word2vec family algorithms.







Setting up N2V
~~~~~~~~~~~~~~

To do. We will submit N2V to PyPI as soon as the code is a bit more mature.



Unit testing
^^^^^^^^^^^^

cd in the N2V directory and enter: ::

    $ nosetests tests/





