.. _rstword2vec:

========
Word2vec
========

node2vec family algorithms generate random walks and pass the series of nodes
visited during the walks to word2vec for embedding. embiggen implements word2vec
using both skip-gram and continuous bag of words. This tutorial page will
demonstrate how to run word2vec using texts.


Getting ready
~~~~~~~~~~~~~
To run this tutorial, you will need a corpus of text. If you do not have anything handy,
you can download a book from `Project Gutenberg <https://www.gutenberg.org/>`_, or a text
dataset from `Kaggle <https://www.kaggle.com/>`_.  In this example, we will use the
set of `Hillary Clinton's emails <https://www.kaggle.com/kaggle/hillary-clinton-emails>`_,
which we found to be a relatively small dataset that gives interesting results.



