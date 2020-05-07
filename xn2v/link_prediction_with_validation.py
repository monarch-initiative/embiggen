import sys
import numpy as np  # type: ignore
from sklearn import metrics  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn import svm  # type: ignore
from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore

from xn2v.utils import load_embeddings


import logging
# import os


# handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "link_prediction.log"))
# formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
# handler.setFormatter(formatter)
# log = logging.getLogger()
# log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
# log.addHandler(handler)

class LinkPredictionWithValidation:
    def __init__(self, pos_train_graph, pos_validation_graph, pos_test_graph, neg_train_graph, neg_validation_graph,
                 neg_test_graph, embedded_train_graph_path, edge_embedding_method, classifier):
        """
        Set up for predicting links from results of node2vec analysis
        :param pos_train_graph: The training graph
        :param pos_validation_graph: The validation graph
        :param pos_test_graph:  Graph of links that we want to predict
        :param neg_train_graph: Graph of non-existence links in training graph
        :param neg_validation_graph: Graph of non-existence links in the validation graph
        :param neg_test_graph: Graph of non-existence links that we want to predict as negative edges
        :param embedded_train_graph_path: The file produced by word2vec with the nodes embedded as vectors
        :param edge_embedding_method: The method to embed edges. It can be "hadamard", "average", "weightedL1" or
            "weightedL2"
        :param classifier: classification method. It can be either "LR" for logistic regression, "RF" for random forest
            or "SVM" for support vector machine
        """
        self.pos_train_edges = pos_train_graph.edges()
        self.pos_test_edges = pos_test_graph.edges()
        self.pos_valid_edges = pos_validation_graph.edges()
        self.neg_train_edges = neg_train_graph.edges()
        self.neg_valid_edges = neg_validation_graph.edges()
        self.neg_test_edges = neg_test_graph.edges()
        self.train_nodes = pos_train_graph.nodes()
        self.validation_nodes = pos_validation_graph.nodes()
        self.test_nodes = pos_test_graph.nodes()
        self.embedded_train_graph = embedded_train_graph_path
        self.map_node_vector = load_embeddings(self.embedded_train_graph)
        self.edge_embedding_method = edge_embedding_method
        self.train_edge_embs = []
        self.valid_edges_embs = []
        self.test_edge_embs = []
        self.train_edge_labels = []
        self.test_edge_labels = []
        self.valid_edge_labels = []
        self.classifier = classifier
        self.validation_predictions = []
        self.test_predictions = []
        self.test_confusion_matrix = []
        self.validation_confusion_matrix = []
        self.valid_edge_embs = None
        self.valid_roc = None
        self.valid_average_precision = None
        self.test_roc = None
        self.test_average_precision = None

    def prepare_training_validation_test_labels(self):
        """
        label positive edge embeddings with 1 and negative edge embeddings with 0.
        :return:
        """
        pos_train_edge_embs = self.transform(edge_list=self.pos_train_edges, node2vector_map=self.map_node_vector)
        neg_train_edge_embs = self.transform(edge_list=self.neg_train_edges, node2vector_map=self.map_node_vector)
        self.train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
        # Create train-set edge labels: 1 = true edge, 0 = false edge
        self.train_edge_labels = np.concatenate([np.ones(len(pos_train_edge_embs)), np.zeros(len(neg_train_edge_embs))])

        # Validation-set edge embeddings, labels
        pos_valid_edge_embs = self.transform(edge_list=self.pos_valid_edges, node2vector_map=self.map_node_vector)
        neg_valid_edge_embs = self.transform(edge_list=self.neg_valid_edges, node2vector_map=self.map_node_vector)
        self.valid_edge_embs = np.concatenate([pos_valid_edge_embs, neg_valid_edge_embs])
        # Create train-set edge labels: 1 = true edge, 0 = false edge
        self.valid_edge_labels = np.concatenate([np.ones(len(pos_valid_edge_embs)), np.zeros(len(neg_valid_edge_embs))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = self.transform(edge_list=self.pos_test_edges, node2vector_map=self.map_node_vector)
        neg_test_edge_embs = self.transform(edge_list=self.neg_test_edges, node2vector_map=self.map_node_vector)
        self.test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
        # Create test-set edge labels: 1 = true edge, 0 = false edge
        self.test_edge_labels = np.concatenate([np.ones(len(pos_test_edge_embs)), np.zeros(len(neg_test_edge_embs))])
        logging.info("get test edge labels")

        logging.info("Training edges (positive): {}".format(len(pos_train_edge_embs)))
        logging.info("Training edges (negative): {}".format(len(neg_train_edge_embs)))
        logging.info("Validation edges (positive): {}".format(len(pos_valid_edge_embs)))
        logging.info("Validation edges (negative): {}".format(len(neg_valid_edge_embs)))
        logging.info("Test edges (positive): {}".format(len(pos_test_edge_embs)))
        logging.info("Test edges (negative): {}".format(len(neg_test_edge_embs)))


    def predict_links(self):
        """
        Train  classifier on train-set edge embeddings. Classifier is LR:logistic regression or RF:random forest
        or SVM:support vector machine. All classifiers work using default parameters.
        :return:
        """

        if self.classifier == "LR":
            edge_classifier = LogisticRegression()
        elif self.classifier == "RF":
            edge_classifier = RandomForestClassifier()
        else:
            # implement linear SVM. #TODO: Use non-default parameters on HPC!
            model_svc = svm.LinearSVC()
            edge_classifier = CalibratedClassifierCV(model_svc)

        edge_classifier.fit(self.train_edge_embs, self.train_edge_labels)


        self.train_predictions = edge_classifier.predict(self.train_edge_embs)
        self.train_confusion_matrix = metrics.confusion_matrix(self.train_edge_labels, self.train_predictions)

        self.validation_predictions = edge_classifier.predict(self.valid_edge_embs)
        self.validation_confusion_matrix = metrics.confusion_matrix(self.valid_edge_labels, self.validation_predictions)

        self.test_predictions = edge_classifier.predict(self.test_edge_embs)
        self.test_confusion_matrix = metrics.confusion_matrix(self.test_edge_labels, self.test_predictions)

        # Predicted edge scores: probability of being of class "1" (real edge)
        train_preds = edge_classifier.predict_proba(self.train_edge_embs)[:, 1]
        validation_preds = edge_classifier.predict_proba(self.valid_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(self.test_edge_embs)[:, 1]
        # fpr, tpr, _ = roc_curve(self.test_edge_labels, test_preds)

        self.train_roc = roc_auc_score(self.train_edge_labels, train_preds)  # get the auc score of training
        self.train_average_precision = average_precision_score(self.train_edge_labels, train_preds)
        self.valid_roc = roc_auc_score(self.valid_edge_labels, validation_preds)  # get the auc score of validation
        self.valid_average_precision = average_precision_score(self.valid_edge_labels, validation_preds)
        self.test_roc = roc_auc_score(self.test_edge_labels, test_preds)  # get the auc score of test
        self.test_average_precision = average_precision_score(self.test_edge_labels, test_preds)

    def predicted_ppi_links(self):
        """
        :return: positive test/validation edges and their prediction, 1: predicted correctly, 0: otherwise
        """

        logging.info("positive validation edges and their prediction:")
        for i in range(len(self.pos_valid_edges)):
            logging.info("edge {} prediction {}".format(self.pos_valid_edges[i], self.validation_predictions[i]))

        logging.info("positive test edges and their prediction:")
        for i in range(len(self.pos_test_edges)):
            logging.info("edge {} prediction {}".format(self.pos_test_edges[i], self.test_predictions[i]))

    def predicted_ppi_non_links(self):
        """
        :return: negative test edges (non-edges) and their prediction, 0: predicted correctly, 1: otherwise
        """
        logging.info("negative validation edges and their prediction:")

        for i in range(len(self.neg_valid_edges)):
            logging.info("edge {} prediction {}".format(self.neg_valid_edges[i], self.validation_predictions[i + len(self.pos_valid_edges)]))

        logging.info("negative test edges and their prediction:")

        for i in range(len(self.neg_test_edges)):
            logging.info("edge {} prediction {}".format(self.neg_test_edges[i], self.test_predictions[i + len(self.pos_test_edges)]))

    def output_classifier_results(self):
        """
        The method prints some metrics of the performance of the logistic regression classifier. including accuracy,
        specificity and sensitivity
        predictions: prediction results of the logistic regression
        confusion_matrix:  confusion_matrix[0, 0]: True negatives, confusion_matrix[0, 1]: False positives,
        confusion_matrix[1, 1]: True positives and confusion_matrix[1, 0]: False negatives
        test_roc: AUC score
        test_average_precision: Average precision
         """
        train_conf_matrix = self.train_confusion_matrix
        total = sum(sum(train_conf_matrix))
        train_accuracy = (train_conf_matrix[0, 0] + train_conf_matrix[1, 1]) / total
        train_specificity = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[0, 1])
        train_sensitivity = train_conf_matrix[1, 1] / (train_conf_matrix[1, 0] + train_conf_matrix[1, 1])
        train_f1_score = (2.0 * train_conf_matrix[1, 1]) / (
                    2.0 * train_conf_matrix[1, 1] + train_conf_matrix[0, 1] + train_conf_matrix[1, 0])
        # f1-score =2 * TP / (2 * TP + FP + FN)

        logging.info("predictions for training set:")
        logging.info("predictions (training): {}".format(str(self.train_predictions)))
        logging.info("confusion matrix (training): {}".format(str(train_conf_matrix)))
        logging.info('Accuracy (validation) : {}'.format(train_accuracy))
        logging.info('Specificity (validation): {}'.format(train_specificity))
        logging.info('Sensitivity (validation): {}'.format(train_sensitivity))
        logging.info('F1-score (validation): {}'.format(train_f1_score))
        logging.info("node2vec Test ROC score (training): {} ".format(str(self.train_roc)))
        logging.info("node2vec Test AP score (training): {} ".format(str(self.train_average_precision)))

        valid_conf_matrix = self.validation_confusion_matrix
        total = sum(sum(valid_conf_matrix))
        valid_accuracy = (valid_conf_matrix[0, 0] + valid_conf_matrix[1, 1]) / total
        valid_specificity = valid_conf_matrix[0, 0] / (valid_conf_matrix[0, 0] + valid_conf_matrix[0, 1])
        valid_sensitivity = valid_conf_matrix[1,1] / (valid_conf_matrix[1, 0] + valid_conf_matrix[1, 1])
        valid_f1_score = (2.0 * valid_conf_matrix[1,1]) / (2.0 * valid_conf_matrix[1,1] + valid_conf_matrix[0, 1] + valid_conf_matrix[1, 0])
        #f1-score =2 * TP / (2 * TP + FP + FN)

        logging.info("predictions for validation set:")
        logging.info("predictions (validation): {}".format(str(self.validation_predictions)))
        logging.info("confusion matrix (validation): {}".format(str(valid_conf_matrix)))
        logging.info('Accuracy (validation) : {}'.format(valid_accuracy))
        logging.info('Specificity (validation): {}'.format(valid_specificity))
        logging.info('Sensitivity (validation): {}'.format(valid_sensitivity))
        logging.info('F1-score (validation): {}'.format(valid_f1_score))

        logging.info("node2vec Test ROC score (validation): {} ".format(str(self.valid_roc)))
        logging.info("node2vec Test AP score (validation): {} ".format(str(self.valid_average_precision)))

        test_conf_matrix = self.test_confusion_matrix
        total = sum(sum(test_conf_matrix))
        test_accuracy = (test_conf_matrix[0,0] + test_conf_matrix[1,1]) / total
        test_specificity = test_conf_matrix[0,0] / (test_conf_matrix[0,0] + test_conf_matrix[0,1])
        test_sensitivity = test_conf_matrix[1,1] / (test_conf_matrix[1,0] + test_conf_matrix[1,1])
        test_f1_score = (2.0 * test_conf_matrix[1,1]) / (2.0 * test_conf_matrix[1,1] + test_conf_matrix[0, 1] + test_conf_matrix[1, 0])

        logging.info("predictions for test set:")
        logging.info("predictions (test): {}".format(str(self.test_predictions)))
        logging.info("confusion matrix (test): {}".format(str(test_conf_matrix)))
        logging.info('Accuracy (test): {}'.format(test_accuracy))
        logging.info('Specificity (test): {}'.format(test_specificity))
        logging.info('Sensitivity (test): {}'.format(test_sensitivity))
        logging.info('F1-score (test): {}'.format(test_f1_score))
        logging.info("node2vec Test ROC score (test): {} ".format(str(self.test_roc)))
        logging.info("node2vec Test AP score (test): {} ".format(str(self.test_average_precision)))

    def transform(self, edge_list, node2vector_map):
        """
        This method finds embedding for edges of the graph. There are 4 ways to calculate edge embedding: Hadamard,
        Average, Weighted L1 and Weighted L2
        :param edge_list:
        :param node2vector_map: key:node, value: embedded vector
        # :param size_limit: Maximum number of edges that are embedded
        :return: list of embedded edges
        """
        embs = []
        edge_embedding_method = self.edge_embedding_method
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = node2vector_map[node1]
            emb2 = node2vector_map[node2]
            if edge_embedding_method == "hadamard":
                # Perform a Hadamard transform on the node embeddings.
                # This is a dot product of the node embedding for the two nodes that
                # belong to each edge
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
                logging.error("Enter hadamard, average, weightedL1, weightedL2")
                sys.exit(1)
            embs.append(edge_emb)
        embs = np.array(embs)
        return embs

    def output_edge_node_information(self):
        self.edge_node_information(self.pos_train_edges, "positive_training")
        self.edge_node_information(self.pos_valid_edges, "positive_validation")
        self.edge_node_information(self.pos_test_edges, "positive_test")

    def edge_node_information(self, edge_list, group):
        """
        print the number of nodes and edges of each type of the graph
        :param edge_list: e.g.,  [('1','7), ('88','22'),...], either training or test
        :param group:
        :return:
        """
        num_edges = 0
        nodes = set()
        for edge in edge_list:
            num_edges += 1
            nodes.add(edge[0])
            nodes.add(edge[1])

        logging.info("##### edge/node diagnostics for {} #####".format(group))
        logging.info("{}: number of  edges : {}".format(group, num_edges))
        logging.info("{}: number of nodes : {}".format(group, len(nodes)))
