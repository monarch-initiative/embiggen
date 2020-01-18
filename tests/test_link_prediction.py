import unittest
from unittest import TestCase
import os

from n2v import LinkPrediction, CSFGraph

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


class TestLinkPrediction(TestCase):

    def test_init(self):
        inputfile = os.path.join(os.path.dirname(
            __file__), 'data', 'small_graph.txt')
        train = CSFGraph(inputfile)
        lp = LinkPrediction(train, train, inputfile, params=params)
        lp.read_embeddings()
        lp.generate_random_negative_edges()
        # lp.predict_links() TODO FIX! This method fails!!
        # lp.output_Logistic_Reg_results() TODO FIX!!!
        # lp.output_diagnostics_to_logfile()  TODO FIX!!!
