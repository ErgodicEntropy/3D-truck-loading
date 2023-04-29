#Truck-based approach

import numpy as np
import random
import math
import itertools
from matplotlib.pyplot import plot as plt

#Decision variables: y[j] = 0,1 (bin usage), x[i][j] = 0,1 (items-bin assignment)
#Objective function = min(sum(y[j] from 1 to n)) worst case scenario: g = n
#Parameters: S[i] sizes of items, B[j] capacities of bins, n number of items, m total number of bins available (not to be confused with number of bins used len(T) = g which is the solution)
#Constraints: sum(S[i]x[i][j])<B[j]y[j] for all j, sum(x[i][j] = 1) for all i, m > len(T), m > n

#First-Fit algorithm (no sorting)

def BP1(n,m,th,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    j = 0
    tabu = np.ones(n)
    while math.floor(B[j]) > th: #Greedy
        for k in range(len(I)):
            if I[k] < B[j] and tabu[k] == 1:
                B[j] = B[j] - I[k]
                tabu[k] = 0
            elif math.floor(B[j]) > th:
                continue
            else:
                break
        j = j+1
    return j+1

# print(BP1(10,20,2,30,14))

#First-Fit decreasing algo (sorting truck capacities): Best-Fit algo

def BP2(n,m,th,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    SB = np.sort(B)[::-1]
    h = 0
    j = int(np.where(B == SB[h])[0])
    tabu = np.ones(n)
    while math.floor(B[j]) > th:
        for k in range(len(I)):
            if I[k] < B[j] and tabu[k] == 1:
                B[j] = B[j] - I[k]
                tabu[k] = 0
            elif math.floor(B[j]) > th:
                continue
            else:
                break
        h = h+1
        j = int(np.where(B == SB[h])[0])
    return h+1

print(BP2(10,20,2,30,14))

