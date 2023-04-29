#Bin packing problem (offline optimization): BP problem is by definition item-based

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

def BP(n,m,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    g = 0 #solution of the problem as a list of used bins (boolean filtration): Memory
    Memory = []
    Bol = True
    for k in range(len(I)):
        for j in range(len(B)):
            for p in range(len(Memory)):
                Bol = Bol*bool(B[j] != Memory[p])
            if I[k] > B[j]:
                pass
            elif Bol == True:
                g = g+1
                B[j] = B[j] - I[k]
                Memory.append(B[j])
                break
            else:
                B[j] = B[j] - I[k]
                break

    return g

print(BP(100,200,10,10))


















#First-Fit decreasing algorihm  (reversed sorting): there exists at least one ordering of items such that the First-Fit algo yields optimal solution (Existence theorem)

def BP(n,m,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    SI = np.sort(I)[::-1]
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    g = 0 #solution of the problem as a list of used bins (boolean filtration)
    Memory = []
    Bol = True
    for k in range(len(SI)):
        for j in range(len(B)):
            for p in range(len(Memory)):
                Bol = Bol*bool(B[j] != Memory[p])
            if SI[k] > B[j]:
                pass
            elif Bol == True:
                g = g+1
                B[j] = B[j] - SI[k]
                Memory.append(B[j])
                break
            else:
                B[j] = B[j] - SI[k]
                break

    return g

print(BP(10,20,15,30))







