#This is not intended to be picked up by pytest
#This script will create the file small_hetrogenous_graph_gene_disease.train that we are using as a training set
"""
g1 g2 10
g1 g3 10
g1 g4 10
(...)
"""


fh = open("data/small_hetrogenous_graph_gene_disease.train", "w")#The generated file needs to be moved to embiggen/data for link prediction

#g1 is connected to g2, g3 ,..., g500
#g2 is connected to g3, g4, ..., g500
#....
# g250 is connected to g251, ...., g500
for j in range(1, 251):
    for i in range(j+1, 501):
        fh.write("g{} g{} 10\n".format(j, i))

#g500 is connected to g501
fh.write("g500 g501 10\n")

#g501 is connected to g502, g503 , ..., g1000
#g502 is connected to g503, g504, ..., g1000
#....
# g750 is connected to g751, ...., g1000
for j in range(501, 751):
    for i in range(j+1, 1001):
        fh.write("g{} g{} 10\n".format(j, i))

#g1, g2 ,..., g250 are connected to d1
for i in range(1, 251):
   fh.write("g{} d1 10\n".format(i))

#g501, g502 ,..., g750 are connected to d2
for i in range(501, 751):
   fh.write("g{} d2 10\n".format(i))

#d1 is onnected to d2, ..., d20
for i in range(1, 21):
   fh.write("d1 d{} 10\n".format(i))

fh.close()

#We expect 1000 gene nodes, 20 disease nodes
