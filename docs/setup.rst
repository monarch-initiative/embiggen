.. _rstsetup:
Setup
=====

n2v is a Python3 package that implements an extension of the node2vec algorithm for heterogeneous networks.

Requirements
~~~~~~~~~~~~
n2v has been tested with Python 3.6 and 3.7. It requires the following packages.

* networkx TODO version...





Obtaining the data
~~~~~~~~~~~~~~~~~~

In the experiments described in the accompanying manuscript, we use
n2v to analyze a heterogeneous network comprised of protein-protein
interactions taken from the `STRING <https://string-db.org/>`_ database,
a gene co-expression network derived from data from the
`Genotype-Tissue Expression (GTEx) <https://gtexportal.org/home/>`_ project,
and a phenotypic similarity matrix derived from the
`Human Phenotype Ontology (HPO) <https://hpo.jax.org/app/>`_ resource.

Download the following files and store them in a subdirectory called
``data`` in the same directory in which ``run_n2v.py`` is located.

* 9606.protein.actions.v11.0.txt.gz (https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz)
* gene2ensembl.gz (ftp://ftp.ncbi.nih.gov/gene/DATA/gene2ensembl.gz)
* GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct.gz (https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct.gz)

You should not unpack these files. If you use a different directory to store the files, you will need to pass the path to this directory to the ``n2vParser`` constructor (see TODO).

We now need to generate the phenotypic similarity file using code from
the `phenol <https://github.com/monarch-initiative/phenol>`_ library. First,
clone an build the library (using git and maven). ::

  $ git clone https://github.com/monarch-initiative/phenol.git
  $ cd phenol
  $ mvn package

We now need to download the following four files.

* hp.obo (http://purl.obolibrary.org/obo/hp.obo)
* phenotype.hpoa (http://compbio.charite.de/jenkins/job/hpo.annotations.2018/lastSuccessfulBuild/artifact/misc_2018/phenotype.hpoa)
* Homo_sapiens.gene_info.gz (ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz)
* mim2gene_medgen ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/mim2gene_medgen

The we run an app included in the phenol distribution as follows (adjust the paths as necessary). ::

  $ java -jar phenol-cli/target/phenol-cli-1.3.4.jar hpo-sim \
  -h hp.obo \
  -a phenotype.hpoa \
  --mimgene2medgen mim2gene_medgen \
  --geneinfo Homo_sapiens.gene_info.gz 

This will generate a file called ``pairwise_disease_similarity.tsv 
`` that should be placed in the ``data`` directory mentioned above. This file has lines such as ::

  gene1   symbol1 gene2   symbol2 similarity
  NCBIGene:23483  TGDS    NCBIGene:5604   MAP2K1  2.5130515079344473
  NCBIGene:23483  TGDS    NCBIGene:59     ACTA2   3.8899923960367837
  NCBIGene:23483  TGDS    NCBIGene:70     ACTC1   2.9499428272247314


We are now ready to run n2v (see :ref:`rstrunning`).

