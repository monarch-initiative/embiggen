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


Getting started
~~~~~~~~~~~~~~~

There are many ways of running python. Here is one recommended way to get this code running. We will remove some of the auxialiary files once the module is about to be submitted, but the following is good for development.  ::

  $ conda deactivate
  $ conda create -n py37 python=3.7
  $ activate py37
  $ pip install tensorflow # currently 2.1
  $ sudo apt install python-nose # if needed
  $ nosetests tests # in N2V directory. 

If the unit tests run correctly, then you are ready to go!




Setting up N2V
~~~~~~~~~~~~~~

To do. We will submit N2V to PyPI as soon as the code is a bit more mature.



Unit testing
^^^^^^^^^^^^
To run the unit testing on the package, generating the coverage and the html report, you can use:

.. code:: bash

    nosetests --with-coverage --cover-package=n2v --cover-html
    # or
    nosetests --nologcover tests/filename.py # suppress log statements
