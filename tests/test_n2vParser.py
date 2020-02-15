# import unittest
# from unittest import TestCase
# import os
#
# from xn2v import xn2vParser
#
# # files used by many tests
# gene2ensembl = os.path.join(os.path.dirname(
#     __file__), 'data', 'small_gene2ensem.txt.gz')
# string_ppi = os.path.join(os.path.dirname(
#     __file__), 'data', 'small_protein_actions_v11.txt.gz')
#
# hpo_path = os.path.join(os.path.dirname(__file__),
#                         'data', 'small_disease_disease.txt')
# gtex_path = os.path.join(os.path.dirname(
#     __file__), 'data', 'small_gtex.txt.gz')
# gene_disease_train_path = os.path.join(
#     os.path.dirname(__file__), 'data', 'small_gene_disease.txt')
# gene_disease_test_path = os.path.join(
#     os.path.dirname(__file__), 'data', 'small_g2d_test.txt')
#
# params = {'gtex_path': gtex_path, 'gene2ensembl_path': gene2ensembl, 'string_path': string_ppi,
#           'g2d_train_path': gene_disease_train_path, 'g2d_test_path': gene_disease_test_path, 'd2d_path': hpo_path}
#
#
# class TestN2vParser(TestCase):
#     def setUp(self) -> None:
#         curdir = os.path.dirname(__file__)
#         self.datadir = os.path.join(curdir, "data")
#     def test_xn2vParser_init(self):
#         p = xn2vParser(data_dir=self.datadir, params=params)
#         p.qc_input()
#
#     def test_get_num_proteins_not_mapped_count(self):
#         p = xn2vParser(data_dir=self.datadir, params=params)
#         # TODO: currently testing only execution, in
#         # the future it will be necessary to also test
#         # if the return value is correct
#         p.get_num_proteins_not_mapped_count()
#
#     def test_get_number_proteins_found_TCRD(self):
#         p = xn2vParser(data_dir=self.datadir, params=params)
#         # TODO: currently testing only execution, in
#         # the future it will be necessary to also test
#         # if the return value is correct
#         p.get_number_proteins_found_TCRD()
#
#     def test_parse(self):
#         p = xn2vParser(data_dir=self.datadir, params=params)
#         # TODO: currently testing only execution, in
#         # the future it will be necessary to also test
#         # if the return value is correct
#         p.parse()
#         p.output_ensembl_gene2gene_id("output.test.txt")
#         os.remove("output.test.txt")
#         p.get_gene_disease_count()
#         p.get_ensembl_gene2gene_map_count()
#         p.get_unique_string_protein_set()
#         p.get_string_valid_ppi_count()
#         p.get_string_raw_ppi_count()
#         p.get_protein2gene_map_count()
#         p.output_nodes_and_edges_test_set("output.tsv")
#         os.remove("output.tsv")
#         p.output_nodes_and_edges("output.tsv")
#         os.remove("output.tsv")
#         p.gene_protein_disease_edges("output.tsv")
#         os.remove("output.tsv")
#
#     def test_summary(self):
#         p = xn2vParser(data_dir=self.datadir, params=params)
#         # TODO: currently testing only execution,
#         # but the following method currently
#         # consists only of a "pass"
#         p.print_summary()
#
#     def test_xn2vParser_wrong_path(self):
#         with self.assertRaises(Exception):
#             xn2vParser(data_dir="tests/dataty", params=params)
#         with self.assertRaises(Exception):
#             xn2vParser(data_dir="tests/test_xn2vParser.py", params=params)
#         with self.assertRaises(Exception):
#             xn2vParser(data_dir="tests/data")
