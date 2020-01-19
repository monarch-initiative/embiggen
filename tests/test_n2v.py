import unittest
from unittest import TestCase
import os

from n2v import WeightedTriple
from n2v import StringInteraction
from n2v import n2vParser

from urllib.request import urlopen
import os.path

# Tale of Two Cities, Dickens, from Gutenberg project
from n2v.text_encoder import TextEncoder
from n2v.word2vec import SkipGramWord2Vec

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


class TestWeightedTriple(TestCase):

    def test_input_types(self):
        with self.assertRaises(TypeError):
            WeightedTriple(42, 84, "bad-weight", "madeup-edgetype")

    def test_get_subject(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        self.assertEqual(42, wt.get_subject())

    def test_get_object(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        assert wt.get_triple() == "42\t84\tmadeup-edgetype"

    def test_create_gene_disease_weighted_triple(self):
        wt1 = WeightedTriple(42, 84, 0.7, "gene_dis")
        wt2 = WeightedTriple.create_gene_disease_weighted_triple(
            42, 84, 0.7
        )
        assert str(wt1) == str(wt2)

    def test_create_gtex(self):
        wt1 = WeightedTriple(42, 84, 0.7, "gtex")
        wt2 = WeightedTriple.create_gtex(
            42, 84, 0.7
        )
        assert str(wt1) == str(wt2)

    def test_map_string_exception(self):
        with self.assertRaises(TypeError):
            WeightedTriple.map_string("Totally not a StringInteraction")

    def test_map_string_to_gene_id_exception(self):
        with self.assertRaises(TypeError):
            WeightedTriple.map_string_to_gene_id(
                "Totally not a StringInteraction", None)

    def test_get_triple(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        self.assertEqual(84, wt.get_object())

    def test_get_edgetype(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        self.assertEqual("madeup-edgetype", wt.get_edgetype())

    def test_get_weight(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        self.assertAlmostEqual(0.7, wt.get_weight(), places=6)

    def test_set_weight(self):
        wt = WeightedTriple(42, 84, 0.7, "madeup-edgetype")
        self.assertAlmostEqual(0.7, wt.get_weight(), places=6)
        wt.set_weight(0.9)
        self.assertAlmostEqual(0.9, wt.get_weight(), places=6)

    def test_create_hpo_weighted_triple(self):
        wt = WeightedTriple.create_hpo_weighted_triple(42, 84, 0.7)
        self.assertEqual(42, wt.get_subject())
        self.assertEqual(84, wt.get_object())
        self.assertEqual("hpo", wt.get_edgetype())
        self.assertAlmostEqual(0.7, wt.get_weight(), places=6)


class TestStringInteraction(TestCase):
    # Display this test line as an array and then combine it with tabs
    # because otherwise the formating will not be the same as in the
    # original STRING file. Note that the fourth column can be empty
    # 0) item_id_a	1) item_id_b 2) mode 3) action
    # 4) a_is_acting 5)score
    threshold = 700
    array1 = ["9606.ENSP00000000233",
              "9606.ENSP00000248901",
              "catalysis",
              "",
              "",
              "f",
              "277"]
    line1 = '\t'.join(array1)
    array2 = ["9606.ENSP00000000233",
              "9606.ENSP00000248901",
              "activation",
              "",
              "",
              "f",
              "277"]
    line2 = '\t'.join(array2)
    array3 = ["9606.ENSP00000000233",
              "9606.ENSP00000248901",
              "activation",
              "activation",
              "",
              "f",
              "309"]

    line3 = '\t'.join(array3)
    array4 = ["9606.ENSP00000000233",
              "9606.ENSP00000248901",
              "activation",
              "",
              "random",
              "f",
              "309"]

    # a line with non-valid triple. "action" = "random" is not valid
    line4 = '\t'.join(array4)

    array5 = ["9606.ENSP00000000233",
              "9606.ENSP00000223369",
              "reaction",
              "",
              "",
              "t",
              "913"]
    # a line with non-valid triple. "action" = "random" is not valid
    line5 = '\t'.join(array5)

    def test1(self):
        self.assertTrue(True)

    # get item_id_a (protein A) from a line
    def test_get_protein_A(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("ENSP00000000233", si1.get_protein_a())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual("ENSP00000000233", si2.get_protein_a())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("ENSP00000000233", si3.get_protein_a())
        si4 = StringInteraction(TestStringInteraction.line4)
        self.assertEqual("ENSP00000000233", si4.get_protein_a())

    # get item_id_b (protein B) from a line
    def test_get_protein_B(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("ENSP00000248901", si1.get_protein_b())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual("ENSP00000248901", si2.get_protein_b())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("ENSP00000248901", si3.get_protein_b())

    # get the mode of the interaction
    def test_get_mode(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("catalysis", si1.get_mode())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual("activation", si2.get_mode())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("activation", si3.get_mode())

    # get the action of the interaction. Action can be "activation", "inhibition" or "".
    def test_get_action(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("", si1.get_action())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual("", si2.get_action())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("activation", si3.get_action())

    # if protein A is acting, it returns True. If not, it returns False. But False does not mean that protein B is acting
    def test_get_a_acting(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual(False, si1.get_a_is_acting())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual(False, si2.get_a_is_acting())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual(False, si3.get_a_is_acting())

    # get the score of the interaction. scores have been multiplied by 1000.
    def test_get_score(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertEqual(277, si1.get_score())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertEqual(277, si2.get_score())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertEqual(309, si3.get_score())

    # We will have a category of "activation" interactions where
    # protein A activates protein B

    def test_is_activation(self):
        si = StringInteraction(TestStringInteraction.line1)
        self.assertFalse(si.is_activation())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertFalse(si2.is_activation())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertTrue(si3.is_activation())

    # We will have a category of "inhibition" interactions where
    # protein A inhibits protein B
    def test_is_inhibition(self):
        si = StringInteraction(TestStringInteraction.line1)
        self.assertFalse(si.is_inhibition())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertFalse(si2.is_inhibition())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertFalse(si3.is_inhibition())

    # We will have a category of "" interactions where
    # action is missing
    def test_is_missing_action(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertTrue(si1.is_missing_action())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertTrue(si2.is_missing_action())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertFalse(si3.is_missing_action())

    def test_is_a_acting(self):
        si1 = StringInteraction(TestStringInteraction.line1)
        self.assertFalse(si1.is_a_acting())
        si2 = StringInteraction(TestStringInteraction.line2)
        self.assertFalse(si2.is_a_acting())
        si3 = StringInteraction(TestStringInteraction.line3)
        self.assertFalse(si3.is_a_acting())

    # We split the name of protein A and get the second part only to write in the output file.
    def test_get_prot_A_pure_name(self):
        si = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("ENSP00000000233", si.get_protein_a_name())

    # We split the name of protein A and get the second part only to write in the output file.
    def test_get_prot_B_pure_name(self):
        si = StringInteraction(TestStringInteraction.line1)
        self.assertEqual("ENSP00000248901", si.get_protein_b_name())

    # We split the name of protein A and get the second part only to write in the output file.
    def test_get_prot_A_pure_name_1(self):
        si = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("ENSP00000000233", si.get_protein_a_name())

    # We split the name of protein A and get the second part only to write in the output file.
    def test_get_prot_B_pure_name_1(self):
        si = StringInteraction(TestStringInteraction.line3)
        self.assertEqual("ENSP00000248901", si.get_protein_b_name())

    # We check if the line has a valid triple (protA, action, protB)
    def test_has_valid_triple(self):
        si = StringInteraction(TestStringInteraction.line3)
        self.assertFalse(si.has_valid_triple(TestStringInteraction.threshold))

    # We check if the line has a valid triple (protA, action, protB)
    def test_has_valid_triple2(self):
        si = StringInteraction(TestStringInteraction.line4)
        self.assertFalse(si.has_valid_triple(TestStringInteraction.threshold))

    def test_has_valid_triple3(self):
        si = StringInteraction(TestStringInteraction.line5)
        self.assertTrue(si.has_valid_triple(TestStringInteraction.threshold))

    # get the triple "["protA", "action", "protB"]"
    def test_get_list_triple1(self):
        si = StringInteraction(TestStringInteraction.line1)
        expected = "['ENSP00000000233', '', 'ENSP00000248901']"
        self.assertEqual(expected, str(si.get_edge_type()))

    # get the triple "["protA", "action", "protB"]"
    def test_get_list_triple2(self):
        si = StringInteraction(TestStringInteraction.line2)
        expected = "['ENSP00000000233', '', 'ENSP00000248901']"
        self.assertEqual(expected, str(si.get_edge_type()))

    # get the triple "["protA", "action", "protB"]"
    def test_get_list_triple3(self):
        si = StringInteraction(TestStringInteraction.line3)
        expected = "['ENSP00000000233', 'activation', 'ENSP00000248901']"
        self.assertEqual(expected, str(si.get_edge_type()))

    def test_execution(self):
        local_file = 'twocities.txt'

        if not os.path.exists(local_file):
            url = 'https://www.gutenberg.org/files/98/98-0.txt'
            with urlopen(url) as response:
                resource = response.read()
                content = resource.decode('utf-8')
                fh = open(local_file, 'w')
                fh.write(content)
        else:
            print("{} was previously downloaded".format(local_file))

        encoder = TextEncoder(local_file)
        data, count, dictionary, reverse_dictionary = encoder.build_dataset()
        model = SkipGramWord2Vec(
            data, worddictionary=dictionary, reverse_worddictionary=reverse_dictionary)
        model.add_display_words(count)
        model.train(display_step=1000)


'''
class Testn2vParser(TestCase):

    def setUp(self):
        self._parser = n2vParser(params=params)
        self._parser.parse()

    def test_not_null(self):
        self.assertIsNotNone(self._parser)

    def test_get_correct_number_of_ensembl_protein_mapping_entries(self):
        # the small_gene2ensem.txt.gz has seven 9606 (human) entries.
        # six of the entries have a ENSP protein id
        self.assertEqual(6, self._parser.get_protein2gene_map_count())

    def test_get_correct_number_of_ensembl_gene_mapping_entries(self):
        # the small_gene2ensem.txt.gz has seven 9606 (human) entries.
        # all of the entries have  ENSG gene id, but there are only
        # four unique gene id's for entries with a protein (other entries
        # are discarded)
        
        self.assertEqual(4, self._parser.get_ensembl_gene2gene_map_count())

    def test_map_gene_ident_gene_id(self):
        self.assertEqual('21', self._parser.ensg2geneID_map.get("ENSG00000167972"))

    def test_above_threshold_ppi(self):
        # zcat small_9606.protein.actions.txt.gz | awk 'BEGIN{FS="\t"} NR>1 $7>699{ lines[$2] = $0 } END { for (l in lines) print lines[l] }' | wc -l

        self.assertEqual(2, self._parser.get_string_raw_ppi_count())

    def test_string_mapping(self):
        pass

    def test_number_unique_string_proteins(self):
        protein_set = self._parser.get_unique_string_protein_set()
        N = len(protein_set)  ## number of unique proteins in STRING dataset
        self.assertEqual(3, N)

    def test_number_gene_disease(self):
        self.assertEqual(22, self._parser.get_gene_disease_count())
'''

if __name__ == '__main__':
    unittest.main()
