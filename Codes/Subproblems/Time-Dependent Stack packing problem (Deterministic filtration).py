#VM packing problem (Stackable code) + delay optimization: deterministic filtration maximization algorithm (O(nlogn)): non-intrinsic TI (variable) where truck selection based on item's arrival time makes sense, no predefined value and compatible with closed-form explicit fitness function (this doesn't minimize the number of trucks used unless a truck-based approach is adopted)

#intrinsicality should be matched with elimination otherwise paradox arises
#In this case, TI is just a placeholder variable for TB

import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt


def BP(n,m,d,E,L,th,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, th hunger threshold
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackibility code of an item i
    LTI = np.zeros(n) # list of late arrival time of each item (days)
    ETI = np.zeros(n) # list of late early time of each item (days)
    for k in range(n):
        I[k] = random.uniform(1,S)
        ETI[k] = random.uniform(1,E)
        LTI[k] = random.uniform(E,L) #Constraint from special relativity
        if k <= d:
            SC[k] = k
        else:
            SC[k] = int(random.uniform(1,d+1))
    VM = [] #stack packing
    S = []
    ST = [] #list of stacks
    for k in range(d):
        S.append(np.where(SC == SC[k]))
        slice = [I[S[k][j]] for j in range(len(S[k]))]
        VM.append(slice)
        ST.append(np.sum(VM[k]))
    SST = np.sort(ST)[::-1]
    B = np.zeros(m) #list of available trucks of length m
    TB = np.zeros(m) # list of arrival time of each truck to plant (days)
    for k in range(m):
        B[k] = random.uniform(1,C)
        TB[k] = random.uniform(1,L)
    TM = np.zeros(m)
    for k in range(m):
        TM[k] = B[k] + TB[k]
    TBM = np.sort(TB)[::-1]
    h = 0
    j = int(np.where(TB == TBM[h])[0])
    F = [[]]
    tabu = np.ones(n)
    while math.floor(B[j]) > th and h <= len(TBM):
        s = []
        for k in range(len(I)):
            if I[k] < B[j] and tabu[k] == 1:
                B[j] = B[j] - I[k]
                tabu[k] = 0
            elif math.floor(B[j]) > th:
                continue
            else:
                break
        h = h+1
        j = int(np.where(TB == TBM[h])[0])

    return h+1




print(BP(200,200,5,1,5,2,14,20))







