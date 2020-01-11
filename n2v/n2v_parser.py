from __future__ import print_function
import gzip
from mailbox import FormatError
import pandas as pd
import os.path
from collections import defaultdict

import logging.handlers
handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "parser.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
log.addHandler(handler)





class WeightedTriple:
    """A class to represent a (subject, predicate, object, weight, edgetype) tuple"""

    def __init__(self, subj,  obj, weight, edgetype):
        self.subject_ = subj
        self.object_ = obj
        self.predicate_ = edgetype
        if not isinstance(weight, (int, float)):
            raise TypeError("weight must be an integer or float")
        self.weight_ = float(weight)

    def get_subject(self):
        return self.subject_

    def get_object(self):
        return self.object_

    def get_edgetype(self):
        return self.predicate_

    def get_weight(self):
        return self.weight_

    def set_weight(self, w):
        self.weight_ = w

    def get_triple(self):
        return "{}\t{}\t{}".format(self.subject_,  self.object_, self.predicate_)

    @staticmethod
    def create_hpo_weighted_triple(disease_a, disease_b, weight):
        return WeightedTriple(disease_a, disease_b, weight, "hpo")

    @staticmethod
    def create_gene_disease_weighted_triple(gene, disease, weight):
        return WeightedTriple(gene, disease, weight, "gene_dis")

    @staticmethod
    def map_string(si):
        if not isinstance(si, StringInteraction):
            raise TypeError("Need to pass a StringInteraction object")
        return WeightedTriple(si.get_protein_a_name(), si.get_protein_b_name(), si.get_score(), "ppi")

    @staticmethod
    def map_string_to_gene_id(si, ens2gene_map):
        """
        Create a WeightedTriple object in which the STRING protein ids (ENSP) have been mapped to the corresponding
        NCBI Gene ids. We assume that the client has checked that both Protein A and B are already in the ens2gene_map.
        If this is not the case, the code will crash.
        :param si: StringInteraction object
        :param ens2gene_map:
        :return: corresponding WeightedTriple object
        """
        if not isinstance(si, StringInteraction):
            raise TypeError("Need to pass a StringInteraction object")
        return WeightedTriple(ens2gene_map.get(si.get_protein_a_name()),
                              ens2gene_map.get(si.get_protein_b_name()), si.get_score(), "ppi")


    @staticmethod
    def protein_a_and_b_in_ens2gene(si, ens2gene_map):
        """
        Checks if the IDs for both protein A and protein B of a STRING interaction line are in our gene2ensembl map
        If not, then we cannot immediatedly use this interaction
        :param si: A StringInteraction object
        :param ens2gene_map:
        :return: True iff both protein IDs are in the gene2ensembl
        """
        if si.get_protein_a_name() not in ens2gene_map.keys():
            return False
        elif si.get_protein_b_name() not in ens2gene_map.keys():
            return False
        return True

    @staticmethod
    def create_gtex(gene_a, gene_b, weight):
        return WeightedTriple(gene_a, gene_b, weight, "gtex")

    def __str__(self):
        return "{}\t{}\t{}\t{}".format(self.subject_,  self.object_, self.weight_, self.predicate_)


class StringInteraction:
    """A class to represent a single protein protein interaction (line in STRING file)"""
    # one line of the String file:
    # protein_A               protein_B              mode         action          a_is_acting     score
    # 9606.ENSP00000000233    9606.ENSP00000248901   activation   activation       f               309

    def __init__(self, line):
        fields = line.split('\t')
        #2017 version has 6 fields, 2019 version has 7 fields
        # The difference affects index 5. For current purposes, we do not care and only need to adjust the
        # index of the last field (which contains the score).
        if len(fields) != 7:
            raise TypeError("Invalid number of fields for STRING interaction.  Are you using a pre-2019 STRING?")
        pro_a = fields[0].split('.')
        if len(pro_a) == 2:
            self.protein_a = pro_a[1]
        else:
            self.protein_a = fields[0]
        pro_b = fields[1].split('.')
        if len(pro_b) == 2:
            self.protein_b = pro_b[1]
        else:
            self.protein_b = fields[1]
        self.mode = fields[2]
        self.action = fields[3]
        # column 4 is directionality, skip here.
        self.a_is_acting = True if fields[5] == "t" else False
        self.score = 0
        try:
            self.score = int(fields[6])
        except:  # TODO: what should we use instead of "except"
            log.debug("protein protein interaction score is not valid {}".format(self.score))
            i = 0
            for f in fields:
                print("{}) {}".format(i, fields[i]))
                i += 1
            raise TypeError("Format error with score field of STRING interaction. Are you using a pre-2019 STRING?")

    def __str__(self):
        return "{}-{} mode: {} action: {} score: {}".format(self.protein_a, self.protein_b, self.mode,
                                                            self.action, self.score)

    def get_protein_a(self):
        return self.protein_a

    def get_protein_b(self):
        return self.protein_b

    def get_mode(self):
        return self.mode

    def get_action(self):
        return self.action

    def get_a_is_acting(self):
        return self.a_is_acting

    def get_score(self):
        return self.score

    #  mode - type of interaction (e.g. "reaction", "expression", "activation",
    #  "ptmod"(post-translational modifications), "binding", "catalysis")
    # action - the effect of the action ("inhibition", "activation" or unknown(missing))
    # a_is_acting - the directionality of the action if applicable ('t' gives that item_id_a is acting upon item_id_b)

    def is_activation(self):
        if self.action == "activation":
            return True
        else:
            return False

    def is_inhibition(self):
        if self.action == "inhibition":
            return True
        else:
            return False

    def is_missing_action(self):
        if self.action == "":
            return True
        else:
            return False

    def is_a_acting(self):
        if self.a_is_acting:
            return True
        else:
            return False



    # removes the first part of the name of protein A, e.g. from "9606.ENSP00000000233" return "ENSP00000000233"
    def get_protein_a_name(self):
        return self.protein_a

    # removes the first part of the name of protein B, e.g. from "9606.ENSP00000000233" return "ENSP00000000233"
    def get_protein_b_name(self):
        return self.protein_b

    def is_score_greater_thresh(self, threshold):
        if self.score >= threshold:
            return True
        else:
            return False

    def get_edge_type(self):
        triple = []
        # edge_type is determined based on
        if self.is_activation():
            edge_type = "activation"
        elif self.is_inhibition():
            edge_type = "inhibition"
        elif self.is_missing_action():
            edge_type = ""
        else:
            raise Exception("action {} is not valid".format(self.action))
        protein_a_pure_name = self.get_protein_a_name()  # get the second part of the name of protein
        protein_b_pure_name = self.get_protein_b_name()  # get the second part of the name of protein
        triple.append(protein_a_pure_name)
        triple.append(edge_type)
        triple.append(protein_b_pure_name)
        return triple

    # a line has a valid triple if action is "activation" or "inhibition" or it is missing " and the score is greater or
    # equal to a threshold"
    def has_valid_triple(self, score_threshold):
        if (self.is_activation() | self.is_inhibition() | self.is_missing_action()) & \
                self.is_score_greater_thresh(score_threshold):
            return True
        else:
            return False


class n2vParser:
    """
    A class to parse the graph.
    """

    def __init__(self, data_dir="data", params={}):

        if os.path.exists(data_dir) and not os.path.isdir(data_dir):
            raise Exception("`data_dir` must point to a directory")
        elif not os.path.exists(data_dir):
            raise Exception("`data_dir` does not exist. Make it and download the required files to run this script.")
        self.data_dir = data_dir
        log.info("Initializing n2vParser with data directory: {}".format(self.data_dir))
        # initialize expected paths to files 

        # initialize dictionaries we will need to parse the data
        self.ensp2geneID_map = {}  # dictionary to map ensemble protein identifier to gene ID
        self.ensg2geneID_map = {}  # key: an ensembl gene is (e.g., ENSG00000227311). Value,
        # and NCBI Gene id (e.g., 2300).
        self.TCRD_prot_geneid_map = {} #key:ensemble protein identifier, value = gene ID
        self.string_ppi = []  # list of interaction objects, still mapped to Ensembl protein IDs
        self.mapped_string = []  # string_ppi
        self.mapped_string2gene = []  # map string_ppi to Gene ids
        self.hpo_sim = []  # list of disease disease similarity objects
        self.gtex = []  # gene-gene expression correlations
        self.gene_disease_training = [] # list of gene disease edges of training wet
        self.gene_disease_test = [] # list of gene disease edges of test set

        self.proteins_not_mapped = set() #list of proteins that were not mpped to gene ids in gene2ensembl
        self.number_proteins_found_geneid_in_TCRD = 0
        # Set up values for parameters
        default_string_ppi_threshold = 700
        default_gtex_threshold = 0.9
        # the following use defaults if params does not contain the item
        self.string_interaction_threshold_ = params.get('string_threshold', default_string_ppi_threshold)
        self.gtex_threshold_ = params.get('gtex_threshold', default_gtex_threshold)

        # Same for files. Note that the user can override the defaults by providing arguments for the paths
        # to these files.
        #latest files
        default_gtex_path = os.path.join(data_dir, "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz")
        default_gene2ensembl_path = os.path.join(data_dir, "gene2ensembl.gz")
        default_string_path = os.path.join(data_dir, "9606.protein.actions.v10.txt.gz")
        # Dictionary of main results to print out for user
        self.summary = defaultdict(str)



        self.gene_expression_path = params.get('gtex_path', default_gtex_path)
        self.gene2ensembl_path = params.get('gene2ensembl_path', default_gene2ensembl_path)
        self.string_ppi_path = params.get('string_path', default_string_path)

        if 'g2d_train_path' in params:
            self.gene_disease_train_path = params.get('g2d_train_path')
        else:
            raise Exception("Need to provide g2d_train file in parameters")

        if 'g2d_test_path' in params:
            self.gene_disease_test_path = params.get('g2d_test_path')
        else:
            raise Exception("Need to provide g2d_test file in parameters")

        if 'd2d_path' in params:
            self.pairwise_disease_sim_path = params.get('d2d_path')
        else:
            raise Exception("Need to provide d2d file in parameters")
        if "TCRD_path" in params:
            self.TCRD_path = params.get("TCRD_path")
        else:
            self.TCRD_path = None

        # check whether the files actually exist. If not, raise an Error
        self.qc_input()

        log.info("[INFO] Parameters:")
        log.info("\tString interaction threshold: {}".format(self.string_interaction_threshold_))
        log.info("\tGTEx correlation threshold: {}".format(self.gtex_threshold_))
            
    def qc_input(self):
        if not os.path.isfile(self.gene2ensembl_path):
            raise Exception("Could not find gene2ensembl file at {}".format(self.gene2ensembl_path))
        else:
            log.info("gene2ensembl: {}".format(self.gene2ensembl_path))
        if not os.path.isfile(self.string_ppi_path):
             raise Exception("Could not find STRING PPI file at {}".format(self.string_ppi_path))
        else:
            log.info("STRING PPI file: {}".format(self.string_ppi_path))
        if not os.path.isfile(self.gene_expression_path):
             raise Exception("Could not find GTEx file at {}".format(self.gene_expression_path))
        else:
            log.info("GTEx file: {}".format(self.gene_expression_path))
        if not os.path.isfile(self.pairwise_disease_sim_path):
            raise Exception("Could not find HPO pairwise disease similarity file at {}".format(self.pairwise_disease_sim_path))
        else:
            print("[INFO] HPO pairwise disease similarity file: {}".format(self.pairwise_disease_sim_path))
        if not os.path.isfile(self.gene_disease_train_path):
            raise Exception("Could not find gene disease training similarity file at {}".format(self.gene_disease_train_path))
        else:
            print("[INFO] gene disease training file: {}".format(self.gene_disease_train_path))

        # if not os.path.isfile(self.gene_disease_test_path):
        #   raise Exception(
        #       "Could not find gene disease test similarity file at {}".format(self.gene_disease_test_path))
        # else:
        #    print("[INFO] gene disease test file: {}".format(self.gene_disease_test_path))
        print("[INFO] All four input files were found")

    def _parse_gene2ensembl(self):
        with gzip.open(self.gene2ensembl_path, 'rt') as f:
            #log.debug("start")
            for line in f:
                if line.startswith('#'):
                    continue  # skip header/comment line
                if not line.startswith('9606'):
                    continue  # 9606 is the human NCBI taxon id. Skip other organisms
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 7:
                    raise Exception("Input line({}) has {} fields instead of 7 as expected".format(line, len(fields)))
                ensembl_protein_identifier = fields[6]
                if ensembl_protein_identifier == " " or not ensembl_protein_identifier.startswith("ENSP"):
                    continue  # no protein ID available for this, skip line
                    # Note that the ensembl ids can have version numbers, e.g., ENSP00000464619.2
                    # We will remove the version numbers so that we can match the STRING data
                    # because the STRING files do not include ensembl version numbers
                fields_ens = ensembl_protein_identifier.split('.')
                ensembl_protein_identifier = fields_ens[0]
                ensembl_gene_id = fields[2]
                ncbi_gene_id = fields[1]
                # Ensembl protein (ENSP) to NCBI Gene ID map
                self.ensp2geneID_map[ensembl_protein_identifier] = ncbi_gene_id
                # Ensembl gene (ENSG) to NCBI Gene ID map
                self.ensg2geneID_map[ensembl_gene_id] = ncbi_gene_id
        log.info("{} protein to gene mappings were extracted from {}."
                 .format(len(self.ensp2geneID_map), self.gene2ensembl_path))

        # The data were extracted from TCRD:
        # CGL_HUMAN P32929 1491 CTH ENSP00000359976
        # CG077_HUMAN A4D0Y5 154872 C7orf77 ENSP00000480627
        # We  map ensembl_protein_identifier to gene id

    def _parse_TCRD(self):
        """
        Parse the mapping file obtained from TCRD. This file contains some additional mappings that
        Are not present in the ensembl2gene file.
        :return:
        """
        genes_in_ensembl = 0
        mismatches = 0
        genes_not_in_ensembl = 0
        line_count = 0
        with open(self.TCRD_path, 'r') as f:
            for line in f:
                line_count += 1
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 5:
                    raise FormatError("Input line({}) has {} fields instead of 5 as expected".format(line, len(fields)))
                protein_identifier = fields[4]
                if protein_identifier.startswith("ENSP"):
                    gene_id = fields[2]
                    if protein_identifier in self.ensp2geneID_map.keys():
                        if self.ensp2geneID_map.get(protein_identifier) == gene_id:
                            # all is good. mapping already there
                            genes_in_ensembl += 1
                        else:
                            log.debug("Gene id in ensembl={} but in TCRD={} for protein {}".format(gene_id, self.ensp2geneID_map.get(protein_identifier), protein_identifier))
                            mismatches += 1
                            continue
                    else:
                        # Add new mapping to map
                        self.ensp2geneID_map[protein_identifier] = gene_id
                        genes_not_in_ensembl += 1
            log.info("{} lines from TCRD data were parsed.".format(line_count))
            log.info("{} mappings: already present. {} mappings: added from TCRD. {} mismatches".format(genes_in_ensembl, genes_not_in_ensembl, mismatches))


    def _parse_unique_valid_prot_b_string_ppi(self):
        """
        If protA and protB have interactions, like
        9606.ENSP00000000233	9606.ENSP00000216366	binding		f	f	165
        9606.ENSP00000000233	9606.ENSP00000216366	reaction		f	f	165
        9606.ENSP00000000233	9606.ENSP00000216366	reaction		t	f	165
        9606.ENSP00000000233	9606.ENSP00000216366	reaction		t	t	165
        we choose the maximum score and then the line with the maximum score is checked whether it is valid or not
        (i.e. maximum score > threshold and mode is either "inhibition", "activation" or missing).
        If it has a valid triple, it is added to the list string_ppi. So, string_ppi contains unique lines,
        ie. for each prot_a and prot_b (if they form a valid triple), there exists one line
        """
        with gzip.open(self.string_ppi_path, 'rt') as f:
            next(f) #skip the header
            line = next(f)
            # Get the very first interaction (second line of the file)
            # and initialize values for previous_score etc.
            string_int_prev = StringInteraction(line)
            previous_score = string_int_prev.get_score()
            previous_prot_a = string_int_prev.get_protein_a_name()
            previous_prot_b = string_int_prev.get_protein_b_name()
            for line in f:
                string_int = StringInteraction(line)
                if string_int.get_protein_a_name() == previous_prot_a and string_int.get_protein_b_name() == previous_prot_b:
                    if string_int.get_score() > previous_score:
                        string_int_prev = StringInteraction(line)
                else:
                    # if we get here, then we are starting a block of lines with a new protein pair
                    # first we enter the previous interaction into the list, and then we update the
                    # value of string_int_prev
                    if string_int_prev.has_valid_triple(self.string_interaction_threshold_):
                        self.string_ppi.append(string_int_prev)
                    string_int_prev = StringInteraction(line)
                previous_score = string_int.get_score()
                previous_prot_a = string_int.get_protein_a_name()
                previous_prot_b = string_int.get_protein_b_name()
            # Add the very last STRING interaction
            if string_int_prev.has_valid_triple(self.string_interaction_threshold_):
                   self.string_ppi.append(string_int_prev)
        log.info("Retrieved n={} protein protein interactions from STRING data".format(len(self.string_ppi)))
        return self.string_ppi

    def _map_string2geneid(self):
        bad_interaction = 0
        c = 0
        log.debug("Starting to _map_string2geneid")
        for i in self.string_ppi:
            if WeightedTriple.protein_a_and_b_in_ens2gene(i, self.ensp2geneID_map):
                wt = WeightedTriple.map_string_to_gene_id(i, self.ensp2geneID_map)
                self.mapped_string2gene.append(wt)
            else:
                bad_interaction += 1
                if i.get_protein_a_name() not in self.ensp2geneID_map.keys():
                    self.proteins_not_mapped.add(i.get_protein_a_name())
                elif i.get_protein_b_name() not in self.ensp2geneID_map.keys():
                    self.proteins_not_mapped.add(i.get_protein_b_name())
            c += 1
            if c % 5000 == 0:
                print("{} STRING interactions".format(c))
        log.info("Mapped {} interactions successfully. "
                 "Could not map {} interactions because of a total of {} "
                 "unmappable proteins".format(len(self.mapped_string2gene), bad_interaction, len(self.proteins_not_mapped)))


    def output_proteins_not_mapped(self): #writing proteins that are not mapped to gene id in gene2ensembl
        # in the proteins_not_mapped.txt file
        output_file = open("unmapped_proteins.txt", "w")
        for prot in self.proteins_not_mapped:
            output_file.write(prot + "\n")
        output_file.close()

    def get_num_proteins_not_mapped_count(self): #number of proteins that are not mapped to gene ids in gene2ensembl
       return len(self.proteins_not_mapped)

    def get_number_proteins_found_TCRD(self): # get the number of proteins that their gene
        # ids were not found in gene2ensembl, but were found in TCRD
         return self.number_proteins_found_geneid_in_TCRD

    def _parse_pairwise_disease_similarity_file(self):
        """
        Parse the n2v/data/pairwise_disease_similarity.tsv file from pairdc
        disease1	d2d	disease2	similarity
        OMIM:212050	d2d	OMIM:308230	3.2942230999469757
        (...)
        """
        with open(self.pairwise_disease_sim_path, 'rt') as f:
            for line in f:
                if line.startswith("OMIM"):
                    fields = line.rstrip('\n').split('\t')
                    if len(fields) != 4:
                        raise FormatError("Input line({}) has {} fields instead of 4 as expected".format(line, len(fields)))
                    disease_a = fields[0].replace("OMIM:", "")
                    disease_b = fields[2].replace("OMIM:", "")
                    weight = float(fields[3])
                    wtrip = WeightedTriple.create_hpo_weighted_triple(disease_a, disease_b, weight)
                    self.hpo_sim.append(wtrip)
                    # print(wtrip)
        log.info("{} hpo disease-disease similarity entries were parsed.".format(len(self.hpo_sim)))


    def _parse_gene_disease_training(self):# parsing gene_disease_training file
        """
        Parse the file of gene to disease lines created by pairdc
        NCBIGene:2972	g2d	OMIM:616202
        NCBIGene:6664	g2d	OMIM:615866
        (...)
        """
        with open(self.gene_disease_train_path, 'r') as f:
            for line in f:
                if not line.startswith("NCBIGene"):
                    continue
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 3:
                    raise FormatError("Input line({}) has {} fields instead of 3 as expected".format(line, len(fields)))
                gene = fields[0].replace("NCBIGene:", "")
                disease = fields[2].replace("OMIM:", "")
                weight = 1000
                wtrip = WeightedTriple.create_gene_disease_weighted_triple(gene, disease, weight)
                self.gene_disease_training.append(wtrip)
               # print(wtrip)
        log.info("{} hpo disease-gene entries were parsed.".format(len(self.gene_disease_training)))

    def _parse_gene_disease_test(self):# parsing gene_disease_test file
        """
        Parse the file of gene to disease lines created by pairdc
        NCBIGene:2972	g2d	OMIM:616202
        NCBIGene:6664	g2d	OMIM:615866
        (...)
        """
        with open(self.gene_disease_test_path, 'r') as f:
            for line in f:
                if not line.startswith("NCBIGene"):
                    continue
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 3:
                    raise FormatError("Input line({}) has {} fields instead of 3 as expected".format(line, len(fields)))
                gene = fields[0].replace("NCBIGene:", "")
                disease = fields[2].replace("OMIM:", "")
                weight = 1000
                wtrip = WeightedTriple.create_gene_disease_weighted_triple(gene, disease, weight)
                self.gene_disease_test.append(wtrip)

        log.info("{} hpo disease-gene (test set) entries were parsed.".format(len(self.gene_disease_test)))

    def _normalize_weights(self, min_weight, max_weight):
        """
        Normalize weights of HPO and GTEx data to 700,1000
        to match the STRING weights
        """
        sum_weights = 0.0
        min_hpo = float('inf')
        max_hpo = float('-inf')
        for wt in self.hpo_sim:
            w = wt.get_weight()
            sum_weights += w
            if min_hpo > w:
                min_hpo = w
            if max_hpo < w:
                max_hpo = w
        for wt in self.hpo_sim:
            w_new = min_weight + (max_weight - min_weight) * (max_hpo - wt.get_weight()) / (max_hpo - min_hpo)
            wt.set_weight(w_new)
        min_gtex = float('inf')
        max_gtex = float('-inf')
        for wt in self.gtex:
            w = wt.get_weight()
            if min_gtex > w:
                min_gtex = w
            if max_gtex < w:
                max_gtex = w
        for wt in self.gtex:
            w_new = min_weight + (max_weight - min_weight) * (max_gtex - wt.get_weight()) / (max_gtex - min_gtex)
            wt.set_weight(w_new)

    def get_protein2gene_map_count(self):
        """
        get the number of valid lines that has all _ensemble gene identifier and gene ID
        """
        return len(self.ensp2geneID_map)

    def get_string_raw_ppi_count(self):
        """
        get the number of valid ppt interactions
        """
        return len(self.string_ppi)

    def get_string_valid_ppi_count(self):
        return len(self.mapped_string)

    def get_unique_string_protein_set(self):
        prots = set()
        for item in self.string_ppi:
            p_a = item.get_protein_a()
            p_b = item.get_protein_b()
            prots.add(p_a)
            prots.add(p_b)
        return prots

    def get_ensembl_gene2gene_map_count(self):
        return len(self.ensg2geneID_map)

    def get_gene_disease_count(self):
        """
        get the number of valid gene_disease edges
        """
        return len(self.gene_disease_training)

    # Output (for debugging)
    def output_ensembl_gene2gene_id(self, outputfile_path):
        output_file = open(outputfile_path, "w")
        output_file.write("Ensembl_gene_identifier " + "\t" + "Gene_ID" + "\n")
        for ens_gene in self.ensg2geneID_map:
            gene_id = self.ensg2geneID_map[ens_gene]
            output_file.write("%s\t%s\n" % (ens_gene, gene_id))
        output_file.close()


    def _parse_gtex(self):
        list_gene_identifiers = []
        gene_exp = pd.read_csv(self.gene_expression_path, sep='\t', header=2)  # read the gene expression data
        gene_exp_data = pd.DataFrame(gene_exp)

        for gene in gene_exp_data["Name"]:
            gene_id_1 = gene.split('.')
            list_gene_identifiers.append(gene_id_1[0])  # gene identifier is in the form of ENSG00000223972.4.
            # So, we get the first part ENSG00000223972

        gene_exp_data.insert(loc=1, column='Gene_ID', value=list_gene_identifiers)  # add a new column
        # to dataset that has the first part of gene ID, (eg. ENSG00000223972)
              
        list_geneid = list(self.ensg2geneID_map.keys())  # get the list of ensemblgene IDs f
        # rom gene2ensembl file. These are gene IDs that we are interested in finding in gtex file and
        # calculating their correlation coefficients
        list_reduced_num_genes = []
        for gene in list_gene_identifiers:
            if gene in list_geneid:  # if gene in gtex file exists in gene2ensembl file,
                # then add it to the list of genes that we want to find its correlation with other genes
                list_reduced_num_genes.append(gene)

        log.info("{} genes found in intersection of STRING and GTEx that will be "
                    "used to calculate correlations ".format(len(list_reduced_num_genes)))
        dataset_intersection_genes = gene_exp_data.loc[gene_exp_data['Gene_ID'].isin(list_reduced_num_genes)]
        # consider a dataset of gtex file corresponding to the genes in list_reduced_num_genes
               
        self.list_genes = list(dataset_intersection_genes["Gene_ID"])  # list_genes is the final list of genes
        gene_expression = dataset_intersection_genes.drop(['Name', 'Gene_ID', "Description"], 1)
        # get the numerical values from data
        gene_expression_tr = gene_expression.T  # transpose of the dataframe to calculate correlations between genes

        corr_matrix = gene_expression_tr.corr()  # calculating correlation coefficients between genes
        # Now output gene pairs which their correlation is above threshold
        for gene_index_1 in range(0, corr_matrix.shape[0]):
            for gene_index_2 in range(gene_index_1 + 1, corr_matrix.shape[1]):
                if corr_matrix.iloc[gene_index_1, gene_index_2] >= self.gtex_threshold_:  # check if the
                    # correlation between two genes is greater than a threshold
                    gene_a = self.list_genes[gene_index_1]
                    gene_b = self.list_genes[gene_index_2]
                    correl = corr_matrix.iloc[gene_index_1, gene_index_2]
                    gene_a_id = self.ensg2geneID_map[gene_a]
                    gene_b_id = self.ensg2geneID_map[gene_b]
                    wt = WeightedTriple.create_gtex(gene_a_id, gene_b_id, correl)
                    self.gtex.append(wt)
        log.info("{} gtex correlations were extracted".format(len(self.gtex)))

    def gene_protein_disease_edges(self, fname):
        # self.mapped_string2gene = []  # map string_ppi to Gene ids
        # self.hpo_sim = []  # list of disease disease similarity objects
        # self.gtex = []  # g
        geneset = set()
        proteinset = set()
        diseaseset = set()
        with open(fname, 'w') as f:
            for i in self.mapped_string2gene:
                sub = i.get_subject()
                obj = i.get_object()
                proteinset.add(sub)
                proteinset.add(obj)
            for i in self.gtex:
                sub = i.get_subject()
                obj = i.get_object()
                geneset.add(sub)
                geneset.add(obj)
            for i in self.hpo_sim:
                sub = i.get_subject()
                obj = i.get_object()
                diseaseset.add(sub)
                diseaseset.add(obj)
            for g in geneset:
                if g in proteinset:
                    # create gene-protein edge
                    f.write("g{}\tp{}\t{}\n".format(g, g, 1000))
            #for prot in proteinset:#TODO: check disease-protein edges
               # gene_id = self.ensembl_protein2gene_map.get(prot)
                #for i in self.gene_disease:
                    #if i.get_subject() == gene_id:
                        #disease = i.get_subject
                         ## create dis-protein edge
                        #f.write("d{}\tp{}\t{}\n".format(disease, prot, 1000))
        f.close()

    def output_nodes_and_edges(self, fname):
        node2nodetype = {}
        with open(fname, 'a') as f:
            for i in self.mapped_string2gene:
                sub = i.get_subject()
                obj = i.get_object()
                wgt = i.get_weight()
                f.write("p{}\tp{}\t{}\n".format(sub, obj, wgt))
                node2nodetype[sub] = 'p'
                node2nodetype[obj] = 'p'
            for i in self.gtex:
                sub = i.get_subject()
                obj = i.get_object()
                wgt = i.get_weight()
                f.write("g{}\tg{}\t{}\n".format(sub, obj, wgt))
                node2nodetype[sub] = 'g'
                node2nodetype[obj] = 'g'
            for i in self.hpo_sim:
                sub = i.get_subject()
                obj = i.get_object()
                wgt = i.get_weight()
                f.write("d{}\td{}\t{}\n".format(sub, obj, wgt))
                node2nodetype[sub] = 'd'
                node2nodetype[obj] = 'd'
            for i in self.gene_disease_training:
                sub = i.get_subject()
                obj = i.get_object()
                wgt = i.get_weight()
                f.write("g{}\td{}\t{}\n".format(sub, obj, wgt))
                node2nodetype[sub] = 'g'
                node2nodetype[obj] = 'd'
        f.close()

    def output_nodes_and_edges_test_set(self, fname):
        """
        :param fname: g2d_association_9_2017_test is the test set which contains gene/disease edges
        :return: the output file contains the geneid and disease and weight which we fixed it to 1000
        """
        node2nodetype = {}
        with open(fname, 'a') as f:

            for i in self.gene_disease_test:
                sub = i.get_subject()
                obj = i.get_object()
                wgt = i.get_weight()
                f.write("g{}\td{}\t{}\n".format(sub, obj, wgt))
                node2nodetype[sub] = 'g'
                node2nodetype[obj] = 'd'
        f.close()

    def parse(self):
        """
        Parse all of the input files for n2v
        """
        log.debug("Parsing gene2ensembl {}".format(self.gene2ensembl_path))
        self._parse_gene2ensembl()
        if self.TCRD_path is not None:
            log.info("Start parsing TCRD file to get ensembl protein identifier and gene id")
            self._parse_TCRD()
        else:
            log.info("Not using TCRD supplemental mapping")
        log.info("Parsing string ({}) to get unique protein b ".format(self.string_ppi_path))
        self._parse_unique_valid_prot_b_string_ppi()
        self._map_string2geneid()
        log.info("Writing proteins that are not mapped to gene id")
        self.output_proteins_not_mapped()
        self.get_number_proteins_found_TCRD()
        log.debug("Number of proteins that were not mapped in gene2ensembl but found their gene ids in TCRD: {}  ".format(self.number_proteins_found_geneid_in_TCRD))

        #log.debug("Start parsing  {} ".format(self.hpo_path))
        self._parse_pairwise_disease_similarity_file()
        log.debug("Parsing gene-disease training file {} ".format(self.gene_disease_train_path))
        self._parse_gene_disease_training()

        log.debug("Parsing gene-disease test file {} ".format(self.gene_disease_test_path))
        self._parse_gene_disease_test()
        log.debug("Parsing gtex files {} ".format(self.gene_expression_path))
        self._parse_gtex()
        self._normalize_weights(self.string_interaction_threshold_, 1000)

    def print_summary(self):
        pass
