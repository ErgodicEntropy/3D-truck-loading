#VM packing problem (Stackable code) + delay optimization: probabilistic filtration First-Fit decreasing algorithm (O(nlogn)): intrinsic TI

import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt


def BP2(n,m,d,E,L,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n)
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackibility code of an item i
    TI = np.zeros(n) # list of arrival time of each item to plant(days)
    LTI = np.zeros(n) # list of late arrival time of each item (days)
    ETI = np.zeros(n) # list of late early time of each item (days)
    for k in range(n):
        I[k] = random.uniform(1,S)
        ETI[k] = random.uniform(1,E)
        LTI[k] = random.uniform(E,L) #Constraint from special relativity
        TI[k] = random.uniform(ETI[k],LTI[k])
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
    g = 0 #solution of the problem as a list of used bins (boolean filtration)
    Memory = []
    Bol = True
    for k in range(len(SST)): #item constraint I1
        th = random.uniform(0,1) #prob threshold accept-reject sampling
        for j in range(len(B)):
            for p in range(len(Memory)):
                Bol = Bol*bool(B[j] != Memory[p])
            if SST[k] > B[j] or int(TI[k]) != int(TB[j]):
                pass
            elif Bol == True and TB[k]/LTI[k] > th:
                g = g+1
                B[j] = B[j] - SST[k]
                Memory.append(B[j])
                break
            else:
                B[j] = B[j] - SST[k]
                break
    return g

print(BP2(1000,25,10,1,5,200,15))

