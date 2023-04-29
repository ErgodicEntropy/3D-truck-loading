#VM packing problem: the version of bin packing problem with the hypothesis that items sharing space is better off than individual loadings (arbitrary vs hierarchical space sharing) : (nuclear mass analogy: weight(stack) < sum(weight of items))

#VM packing problem is about entropy minimization

#1st case: piping (solve item-stack problem and pipe it to stack-truck problem) such that there is an arbitrary cyclicity between stacks and trucks: no stackibility code
#2nd case: Arbitrary cyclicity between items and stacks if there is a stackibility code (number of stacks is predefined)
import numpy as np
import random
import math
import itertools
from matplotlib.pyplot import plot as plt

#Decision variables: y[j] = 0,1 (bin usage), x[i][j] = 0,1 (items-bin assignment)
#Objective function = min(sum(y[j] from 1 to n)) worst case scenario: g = n
#Parameters: S[i] sizes of items, B[j] capacities of bins, n number of items, m total number of bins available (not to be confused with number of bins used len(T) = g which is the solution)
#Constraints: sum(S[i]x[i][j])<B[j]y[j] for all j, sum(x[i][j] = 1) for all i, m > len(T), m > n

#For both cases, we use the First-Fit algorithm/F-F decreasing algorithm (just like we did in bin packing problem)

#Case 1 (no stackibility code) : reward-punishment driven approach

#First-Fit algorithm

def BP1(n,v,m,K,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    S = np.zeros(v) #list of available stacks of length v
    for k in range(v):
        S[k] = random.uniform(1,K)
    B = np.zeros(m) #list of available trucks of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    s = 0 #solution of the problem as a list of used stacks (boolean filtration): MemoryS
    MemoryS = [] #Item-Stack optimization (F-F algo)
    Bol1 = True
    for k in range(len(I)):
        for j in range(len(S)):
            for p in range(len(MemoryS)):
                Bol1 = Bol1*bool(S[j] != MemoryS[p])
            if I[k] > S[j]:
                pass
            elif Bol1 == True:
                s = s+1
                S[j] = S[j] - I[k]
                MemoryS.append(S[j])
                break
            else:
                S[j] = S[j] - I[k]
                break

    t = 0 #solution of the problem as a list of used stacks (boolean filtration): MemoryT
    MemoryT = [] #Stack-Truck optimization (F-F algo)
    Bol2 = True
    for k in range(len(MemoryS)):
        for j in range(len(B)):
            for p in range(len(MemoryT)):
                Bol2 = Bol2*bool(B[j] != MemoryT[p])
            if MemoryS[k] > B[j]:
                pass
            elif Bol2 == True:
                t = t+1
                B[j] = B[j] - MemoryS[k]
                MemoryT.append(B[j])
                break
            else:
                B[j] = B[j] - MemoryS[k]
                break
    return t, MemoryT
print(BP1(200,50,20,20,5,20))

#First-Fit decreasing algorithm

def BP1(n,v,m,K,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    for k in range(n):
        I[k] = random.uniform(1,S)
    SI = np.sort(I)[::-1]
    S = np.zeros(v) #list of available stacks of length v
    for k in range(v):
        S[k] = random.uniform(1,K)
    B = np.zeros(m) #list of available trucks of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    s = 0 #solution of the problem as a list of used stacks (boolean filtration): MemoryS
    MemoryS = [] #Item-Stack optimization (F-F decreasing algo)
    Bol1 = True
    for k in range(len(SI)):
        for j in range(len(S)):
            for p in range(len(MemoryS)):
                Bol1 = Bol1*bool(S[j] != MemoryS[p])
            if SI[k] > S[j]:
                pass
            elif Bol1 == True:
                s = s+1
                S[j] = S[j] - SI[k]
                MemoryS.append(S[j])
                break
            else:
                S[j] = S[j] - SI[k]
                break
    MemoryS2 = np.sort(MemoryS)[::-1]
    t = 0 #solution of the problem as a list of used stacks (boolean filtration): MemoryT
    MemoryT = [] #Stack-Truck optimization (F-F decreasing algo)
    Bol2 = True
    for k in range(len(MemoryS2)):
        for j in range(len(B)):
            for p in range(len(MemoryT)):
                Bol2 = Bol2*bool(B[j] != MemoryT[p])
            if MemoryS2[k] > B[j]:
                pass
            elif Bol2 == True:
                t = t+1
                B[j] = B[j] - MemoryS2[k]
                MemoryT.append(B[j])
                break
            else:
                B[j] = B[j] - MemoryS2[k]
                break
    return t, MemoryT
print(BP1(200,50,20,20,5,20))



#Case 2: stackability code

#First-Fit algo
def BP2(n,m,d,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n)
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackibility code of an item i
    for k in range(n):
        I[k] = random.uniform(1,S)
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
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    g = 0 #solution of the problem as a list of used bins (boolean filtration)
    Memory = []
    Bol = True
    for k in range(len(ST)):
        for j in range(len(B)):
            for p in range(len(Memory)):
                Bol = Bol*bool(B[j] != Memory[p])
            if ST[k] > B[j]:
                pass
            elif Bol == True:
                g = g+1
                B[j] = B[j] - ST[k]
                Memory.append(B[j])
                break
            else:
                B[j] = B[j] - ST[k]
                break
    return g

print(BP2(100,20,5,15,2000))

#First-Fit decreasing algo
def BP2(n,m,d,S,C): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n)
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackibility code of an item i
    for k in range(n):
        I[k] = random.uniform(1,S)
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
    B = np.zeros(m) #list of available bins of length m
    for k in range(m):
        B[k] = random.uniform(1,C)
    g = 0 #solution of the problem as a list of used bins (boolean filtration)
    Memory = []
    Bol = True
    for k in range(len(SST)):
        for j in range(len(B)):
            for p in range(len(Memory)):
                Bol = Bol*bool(B[j] != Memory[p])
            if SST[k] > B[j]:
                pass
            elif Bol == True:
                g = g+1
                B[j] = B[j] - SST[k]
                break
            else:
                B[j] = B[j] - SST[k]
                break
            Memory.append(B[j])
    return g

print(BP2(100,20,5,15,2000))
