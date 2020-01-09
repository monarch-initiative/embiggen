import n2v
from n2v import CSFGraph
from n2v import LinkPrediction
import os


test_file = '/home/robinp/PycharmProjects/IDG2KG-project-management/hn2v/g2d_test.txt'
test_graph = CSFGraph(test_file)


training_file = '/home/robinp/PycharmProjects/IDG2KG-project-management/hn2v/edgelist.txt'
training_graph = CSFGraph(training_file)


parameters = {'edge_embedding_method': "hadamard",
              'portion_false_edges': 1}

path_to_embedded_graph = 'disease.embedded'
lp = LinkPrediction(training_graph, test_graph, path_to_embedded_graph, params=parameters)

#lp.output_diagnostics_to_logfile()
lp.predict_links()
lp.output_Logistic_Reg_results()
#test_file= os.path.join(os.path.dirname(__file__), 'tests', 'data', 'karate.test')


training_graph = CSFGraph(training_file)