MMB_by_MMB.py
Main Function: MMB_by_MMB(Data, target, alpha, p, maxK, verbose = False)

# Input arguments:
# Data: data observation, datatype: ndarray
# target: target variable, datatype: int
# alpha: confidence threshold
# p: number of observation data nodes, datatype: int
# maxK: the maximal degree of any variable, datatype: int

# Output arguments:
# P : Set of parent nodes of the target variable, datatype: ndarray
# C : Set of children nodes of the target variable, datatype: ndarray
# dis_depth1 : Set of  districts nodes of the target variable, datatype: ndarray
# un : Set of nodes connected by undirected edges with the target variable, datatype: ndarray
# ci_test : The number of conditional independence tests in the program, datetype: int

Package requirements:
numpy
math
scipy
itertools