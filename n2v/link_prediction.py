import sys

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import logging
import os
import itertools as IT
import random

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "link_prediction.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
log.addHandler(handler)

class LinkPrediction:
    def __init__(self, train_edges, test_edges, embedded_train_graph_path, params={}):
        """
        Set up for predicting links from results of node2vec analysis
        :param train_edges: The complete training graph
        :param test_edges:  List of links that we want to predict
        :param embedded_train_graph_path: THe file produced by word2vec with the nodes embedded as vectors
        """
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.embedded_train_graph = embedded_train_graph_path
        self.map_node_vector = {}
        self.read_embedded_training_graph()
        default_edge_embedding_method = "hadamard"
        self.edge_embedding_method = params.get('edge_embedding_method', default_edge_embedding_method)

        # for false_edges, We want portion_false_edges times the number of true edges,
        # where portion_false_edges is a number between 1 and 10
        default_portion_false_edges = 2
        self.portion_false_edges = params.get('portion_false_edges', default_portion_false_edges)
        self.false_test_edges = []
        self.false_train_edges = []
        self.true_test_edges = []

    def read_embedded_training_graph(self):
        """
        reading the embedded training graph and mapping each node to a vector
        :return:
        """
        n_lines = 0
        map_node_vector = {}  # reading the embedded graph to a map, key:node, value:vector
        with open(self.embedded_train_graph, 'r') as f:
            for line in f:
                fields = line.split('\t') #the format of each line: node v_1 v_2 ... v_d where v_i's are elements of
                # the array corresponding to the embedding of the node
                embe_vec = [float(i) for i in fields[1:]]
                map_node_vector.update({fields[0]: embe_vec})#map each node to its corresponding vector
                n_lines += 1
        f.close()
        self.map_node_vector = map_node_vector
        log.debug("Finished ingesting {} lines (vectors) from {}".format(n_lines, self.embedded_train_graph))

    def __setup_training_and_test_data(self):

        # true test edges are edges that exist in g_test. These are edges that we would like to predict.
        # Therefore, we can call them false edges.
        self.true_test_edges = [edge for edge in self.test_edges]

        # false edges are edges that don't exit in the training graph and test graph.
        # To find false edges, we first find false edges of training graph and then remove test
        # edges from false edges of the training graph
        log.debug('getting non existing (false) edges')
        #############################
        #############  HERE
        train_nodes = set()
        for edge in self.train_edges:
             train_nodes.add(edge[0])
             train_nodes.add(edge[1])

        for pair in IT.combinations(train_nodes, 2):
            print("pair: ", pair)
        false_training_edges = [pair for pair in IT.combinations(train_nodes, 2)
                                if not pair in self.train_edges]
        log.debug("number of edges that don't exist in training edges but may exist in test edges".
                  format(len(false_training_edges)))
        # The following edges are not either in the training or in the test set.
        self.false_edges = [pair for pair in false_training_edges
                            if not pair in self.test_edges]
        log.debug("number of false edges:{}".format(len(self.false_edges)))

        # create false edges


        number_true_train_edges = len(self.train_edges)

        # We want K (portion_false_edges) times the number of positive edges, where K is a number between 1 and 10

        number_false_train_edges = self.portion_false_edges * number_true_train_edges #We need to carefully choose it!
        number_false_test_edges = self.portion_false_edges * len(self.true_test_edges)#We need to carefully choose it!
        if number_false_train_edges > len(self.false_edges):#for now! It needs to be checked!!
            number_false_train_edges = number_true_train_edges

        if number_false_test_edges > len(self.false_edges):#for now! It needs to be checked!!
            number_false_test_edges = len(self.true_test_edges)
        log.debug("number of false training edges:{}".format(number_false_train_edges))
        log.debug("number of false test edges:{}".format(number_false_test_edges))

        number_rand_numbers = number_false_train_edges + number_false_test_edges
        rand_numbers = self.rand_num_generator(0, number_rand_numbers, number_rand_numbers)#generate random numbers
        #between 0 and number_rand_numbers. rand_numbers gives of shuffling of total indices
        for i in range(0, number_false_train_edges):
            self.false_train_edges.append(self.false_edges[rand_numbers[i]])

        for i in range(number_false_train_edges, number_rand_numbers):
            self.false_test_edges.append(self.false_edges[rand_numbers[i]])



    def predict_links(self):
        self.__setup_training_and_test_data()
        #Train-set edge embeddings and labels
        true_train_edge_embs = self.transform(edge_list=self.train_edges, node2vector_map=self.map_node_vector,
                                              size_limit=len(self.train_edges), edge_embedding_method=self.edge_embedding_method)
        #get the edge embedding of false training edges
        false_train_edge_embs = self.transform(edge_list=self.false_train_edges, node2vector_map=self.map_node_vector,
                                               size_limit=len(self.train_edges), edge_embedding_method=self.edge_embedding_method)
        print(len(true_train_edge_embs),len(false_train_edge_embs))
        train_edge_embs = np.concatenate([true_train_edge_embs, false_train_edge_embs])
        # Create train-set edge labels: 1 = true edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(true_train_edge_embs)), np.zeros(len(false_train_edge_embs))])

        # Test-set edge embeddings and labels
        true_test_edge_embs = self.transform(edge_list=self.true_test_edges, node2vector_map=self.map_node_vector,
                                            size_limit=len(self.true_test_edges), edge_embedding_method=self.edge_embedding_method)

        false_test_edge_embs = self.transform(edge_list=self.false_test_edges, node2vector_map=self.map_node_vector,
                                                     size_limit=len(self.true_test_edges),edge_embedding_method=self.edge_embedding_method)
        test_edge_embs = np.concatenate([true_test_edge_embs, false_test_edge_embs])
        # Create test-set edge labels: 1 = true edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(true_test_edge_embs)), np.zeros(len(false_test_edge_embs))])
        log.debug('get test edge labels')

        #log.debug("Total nodes: {}".format(self.train_edges.number_of_nodes()))
        log.debug("Number of true training edges: {}".format(len(self.train_edges)))
        log.debug("Number of false training edges: {}".format(len(false_train_edge_embs)))
        log.debug("Number of true test edges: {}".format(len(self.true_test_edges)))
        log.debug("Number of false test edges: {}".format(len(false_test_edge_embs)))

        log.debug('logistic regression')
        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression()
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        self.predictions = edge_classifier.predict(test_edge_embs)
        self.confusion_matrix = metrics.confusion_matrix(test_edge_labels, self.predictions)

        # Predicted edge scores: probability of being of class "1" (real edge)
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        fpr, tpr, _ = roc_curve(test_edge_labels, test_preds)

        self.test_roc = roc_auc_score(test_edge_labels, test_preds)#get the auc score
        self.test_average_precision = average_precision_score(test_edge_labels, test_preds)

    def output_Logistic_Reg_results(self):
        """
        The method prints some metrics of the performance of the logistic regression classifier. including accuracy, specificity and sensitivity
        :param predictions: prediction results of the logistic regression
        :param confusion_matrix:  confusion_matrix[0, 0]: True negatives, confusion_matrix[0, 1]: False positives,
        confusion_matrix[1, 1]: True positives and confusion_matrix[1, 0]: False negatives
        :param test_roc: AUC score
        :param test_average_precision: Average precision
        :return:
         """
        confusion_matrix = self.confusion_matrix
        total = sum(sum(confusion_matrix))
        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) * (1.0) / total
        specificity = confusion_matrix[0, 0] * (1.0) / (confusion_matrix[0, 0] + confusion_matrix[0, 1]) * (1.0)
        sensitivity = confusion_matrix[1, 1] * (1.0) / (confusion_matrix[1, 0] + confusion_matrix[1, 1]) * (1.0)

        log.info("predictions: {}".format(str(self.predictions)))
        log.info("confusion matrix: {}".format(str(confusion_matrix)))
        log.debug('Accuracy : {}'.format(accuracy))
        log.debug('Specificity : {}'.format(specificity))
        log.debug('Sensitivity : {}'.format(sensitivity))
        log.debug("het_node2vec Test ROC score: {} ".format(str(self.test_roc)))
        log.debug("het_node2vec Test AP score: {} ".format(str(self.test_average_precision)))

    def transform(self,edge_list, node2vector_map, size_limit, edge_embedding_method):
        """
        This method finds embedding for edges of the graph. There are 4 ways to calculate edge embedding: Hadamard, Average, Weighted L1 and Weighted L2
        :param edge_list:
        :param node2vector_map: key:node, value: embedded vector
        :param size_limit: Maximum number of edges that are embedded
        :param edge_embedding_method:Hadamard, Average, Weighted L1 or Weighted L2
        :return: list of embedded edges
        """
        embs = []
        num_edges = 0
        max_num_edges = size_limit
        for edge in edge_list:
            if num_edges < max_num_edges:
                node1 = edge[0]
                mytype = type(node1)
                print("Node 1 %s: %s" % (node1, mytype))
                node2 = edge[1]
                mytype = type(node2)
                print("Node 2  %s: %s" % (node2, mytype))
                emb1 = node2vector_map[node1]
                emb2 = node2vector_map[node2]
                if edge_embedding_method == "hadamard":
                #Perform a Hadamard transform on the node embeddings.
                #This is a dot product of the node embedding for the two nodes that
                #belong to each edge
                    edge_emb = np.multiply(emb1, emb2)
                elif edge_embedding_method == "average":
                # Perform a Average transform on the node embeddings.
                # This is a elementwise average of the node embedding for the two nodes that
                # belong to each edge
                    edge_emb = np.add(emb1, emb2) / 2
                elif edge_embedding_method == "weightedL1":
                    # Perform weightedL1 transform on the node embeddings.
                    # WeightedL1 calculates the absolute value of difference of each element of the two nodes that
                    # belong to each edge
                    edge_emb = abs(emb1 - emb2)
                elif edge_embedding_method == "weightedL2":
                    # Perform weightedL2 transform on the node embeddings.
                    # WeightedL2 calculates the square of difference of each element of the two nodes that
                    # belong to each edge
                    edge_emb = np.power((emb1 - emb2), 2)
                else:
                    log.error("You need to enter hadamard, average, weightedL1, weightedL2")
                    sys.exit(1)
                embs.append(edge_emb)
                num_edges += 1
            else:
                 break
        embs = np.array(embs)
        return embs

    def output_diagnostics_to_logfile(self):
        self.log_edge_node_information(self.train_edges, "training")
        self.log_edge_node_information(self.test_edges, "test")

    def log_edge_node_information(self, edge_list, group):#TODO:Needs to be modified for homogenous graphs
        """
        log the number of nodes and edges of each type of a hetrogenous graph which has 3 types of nodes: genes, proteins and diseases.
        :param edge_list: e.g.,  [('g1','g7), ('g88','d22'),...], either training or test
        :return:
        """
        num_gene_gene = 0
        num_gene_dis = 0
        num_gene_prot = 0
        num_prot_prot = 0
        num_prot_dis = 0
        num_dis_dis = 0
        num_gene = 0
        num_prot = 0
        num_dis = 0
        nodes = set()
        for edge in edge_list:
            if (edge[0].startswith("g") and edge[1].startswith("g")):
                num_gene_gene += 1
            elif ((edge[0].startswith("g") and edge[1].startswith("d")) or
                  (edge[0].startswith("d") and edge[1].startswith("g"))):
                num_gene_dis += 1
            elif ((edge[0].startswith("g") and edge[1].startswith("p")) or
                  (edge[0].startswith("p") and edge[1].startswith("g"))):
                num_gene_prot += 1
            elif edge[0].startswith("p") and edge[1].startswith("p"):
                num_prot_prot += 1
            elif (edge[0].startswith("p") and edge[1].startswith("d")) or (
                    edge[0].startswith("d") and edge[1].startswith("p")):
                num_prot_dis += 1
            elif edge[0].startswith("d") and edge[1].startswith("d"):
                num_dis_dis += 1
            nodes.add(edge[0])
            nodes.add(edge[1])
        for node in nodes:
            if node.startswith("g"):
                num_gene += 1
            elif node.startswith("p"):
                num_prot += 1
            elif node.startswith("d"):
                num_dis += 1
        log.debug("##### edge/node diagnostics for {} #####".format(group))
        log.debug("{}: number of gene-gene edges : {}".format(group, num_gene_gene))
        log.debug("{}: number of gene-dis edges : {}".format(group, num_gene_dis))
        log.debug("{}: number of gene-prot edges : {}".format(group, num_gene_prot))
        log.debug("{}: number of prot_prot edges : {}".format(group, num_prot_prot))
        log.debug("{}: number of prot_dis edges : {}".format(group, num_prot_dis))
        log.debug("{}: number of dis_dis edges : {}".format(group, num_dis_dis))
        log.debug("{}: number of gene nodes : {}".format(group, num_gene))
        log.debug("{}: number of protein nodes : {}".format(group, num_prot))
        log.debug("{}: number of disease nodes : {}".format(group, num_dis))
        log.debug("##########")

    def rand_num_generator(self, start, end, num):
        """
        The method to generate random numbers between a starting point and end point
        :param start:starting range
        :param end:ending range
        :param num:number of elements needs to be appended
        :return: list of num randomly generated numbers between start and end
        """
        res = []
        for j in range(num):
             res.append(random.randint(start, end))
        return res