import unittest
from unittest import TestCase
import os

from n2v import n2vParser

# files used by many tests
gene2ensembl = os.path.join(os.path.dirname(
    __file__), 'data', 'small_gene2ensem.txt.gz')
string_ppi = os.path.join(os.path.dirname(
    __file__), 'data', 'small_protein_actions_v11.txt.gz')

hpo_path = os.path.join(os.path.dirname(__file__),
                        'data', 'small_disease_disease.txt')
gtex_path = os.path.join(os.path.dirname(
    __file__), 'data', 'small_gtex.txt.gz')
gene_disease_train_path = os.path.join(
    os.path.dirname(__file__), 'data', 'small_gene_disease.txt')
gene_disease_test_path = os.path.join(
    os.path.dirname(__file__), 'data', 'small_g2d_test.txt')

params = {'gtex_path': gtex_path, 'gene2ensembl_path': gene2ensembl, 'string_path': string_ppi,
          'g2d_train_path': gene_disease_train_path, 'g2d_test_path': gene_disease_test_path, 'd2d_path': hpo_path}


class TestN2vParser(TestCase):
    def test_n2vParser(self):
        p = n2vParser(data_dir="tests/data", params=params)

    def test_n2vParser_wrong_path(self):
        with self.assertRaises(Exception):
            n2vParser(data_dir="tests/dataty", params=params)
        with self.assertRaises(Exception):
            n2vParser(data_dir="tests/test_n2vParser.py", params=params)
        with self.assertRaises(Exception):
            n2vParser(data_dir="tests/data")
