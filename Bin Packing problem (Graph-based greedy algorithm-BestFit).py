#Graph-based greedy approach: space residual sorting using a pseudo-complete bipartite graph nxm that is dynamically updated (thus pseudo) => Best-Fit algo
#Luck condition (when greedy approach = dynamic prog) : a condition that ensures the optimality of the greedy approach thus eliminating initial conditions dependence

import numpy as np
import random
import math
import itertools
from matplotlib.pyplot import plot as plt

def BP(n,m,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    R = np.zeros((n,m)) #Residual-weighted pseudo-complete bipartite graph nxm
    S = []
    for j in range(n):
        for k in range(m):
            R[j][k] = B[k] - I[j]
    S.append(R[0].tolist().index(np.max(R[0])))
    for j in range(1,n):
        for k in range(j):
            R[j][S[k]] = R[j][S[k]] - B[S[k]]
        S.append(R[j].tolist().index(np.max(R[j])))
    g = 1
    Bol = True
    for j in range(1,n):
        for k in range(j):
            Bol = Bol* bool(R[j][S[k]] != np.max(R[j]))
        if Bol == True:
            g = g+1
        else:
            pass
    h = [] #persistance/degree of luck (extent to which greedy algo = dynamic prog)
    for k in range(len(S)):
        h.append(S.count(S[k]))
    return h, g

print(BP(15,20,14,19))




