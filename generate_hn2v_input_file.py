# Can be run from ipython as %run  run_hnxn2v_analysis.py
import sys
import os
from xn2v import xn2vParser


import logging.handlers
handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "parser.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
log.addHandler(handler)



data_download_dir = "data"
# 1) Check the required files are present in data directory
def check_existence(url, local_filepath):
    if not os.path.isfile(local_filepath):
        log.error("could file not find at {}".format(local_filepath))
        log.error("[FATAL] Need to download file from {} and place the file in the directory.".format( url))
        sys.exit(1)
    else:
        log.info("Using file {}".format(local_filepath))

string_protein_actions_url = "https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz"
string_local_filename = "9606.protein.actions.v11.0.txt.gz"
string_path = os.path.join(data_download_dir, string_local_filename)
check_existence(string_protein_actions_url, string_path)

gene2ensembl_url = "ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz"
gene2ensembl_file = "gene2ensembl.gz"
gene2ensembl_path = os.path.join(data_download_dir, gene2ensembl_file)
check_existence(gene2ensembl_url, gene2ensembl_path)

gtex_url = "https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
gtex_file = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
gtex_path = os.path.join(data_download_dir, gtex_file)
print("\'{}\'".format(gtex_path))
check_existence(gtex_url, gtex_path)

TCRD_path = os.path.join(data_download_dir, "stringmap_from_TCRD.txt")

g2d_training_path = os.path.join(data_download_dir, "g2d_associations_training_4_2014.tsv")
if not os.path.isfile(g2d_training_path):
    log.error("You need to run pairdc and put its output file (e.g., g2d_associations_training_6_2014.tsv) into the data directory.")
    log.error("Could not find {}".format(g2d_training_path))
    sys.exit(1)

g2d_test_path = os.path.join(data_download_dir, "g2d_associations_test_9_2017.tsv")
if not os.path.isfile(g2d_test_path):
    log.error("You need to run pairdc and put its output file (e.g., g2d_associations_test_9_2017.tsv) into the data directory.")
    log.error("Could not find {}".format(g2d_test_path))
    sys.exit(1)

d2d_path = os.path.join(data_download_dir, 'pairwise_disease_similarity_9_2017.tsv')
if not os.path.isfile(d2d_path):
    log.error("You need to run pairdc and put its output file (e.g., pairwise_disease_similarity.tsv) into the data directory.")
    log.error("Could not find {}".format(d2d_path))
    sys.exit(1)

parameters = {'string_threshold': 700,
              'gtex_threshold': 0.9,
              'gtex_path': gtex_path,
              'gene2ensembl_path': gene2ensembl_path,
              'string_path': string_path,
              'tcrd_path': TCRD_path,
              'g2d_train_path': g2d_training_path,
              'g2d_test_path' : g2d_test_path,
              'd2d_path': d2d_path}

parser = xn2vParser(params=parameters)
parser.parse()
parser.print_summary()
outfilename_training = "edgelist.txt"
parser.gene_protein_disease_edges(outfilename_training)
parser.output_nodes_and_edges(outfilename_training)
outfilename_test = "g2d_test.txt"
parser.output_nodes_and_edges_test_set(outfilename_test)

