#Genetic algorithms with boolean representation to time-dependent item packing-truck usage problem: Minimizing the number of trucks while also selecting the trucks with the maximum ratio arrival-time/latest arrival time

#Methylation Hypothesis: To avoid computational redundancy, it's fair to assume that each truck has a pre-defined number of items K = n/m in it (to be known). This hypothesis treats the item-truck insertion operation as a black box aposteriori (meaning it already happened without knowing how)
#Corollary of this assumption: Genetic algorithm will be applied on y[j]s only

#Genetic algorithm: population generated are divided into two types of chromosomes (almost two different populations but linked in an order-specific ) : K-lists (e population of lists with K 1s each) related to the first termm of the objective function (a*sum(TCt*y)), and p-lists (p-truncated of m! population of permuted lists for each K-list). Therefore, Npop = ep ((e x m)! without truncation) where Npop is the total number of solutions in the population.

import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt

def factorial(N):
    if N == 0:
        return 1
    elif N == 1:
        return N
    else:
        return N*factorial(N-1)

def BP(n,m,a,b,TC,ICM,E,L,S,C,p,e,MaxIter): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, TC biggest truck cost, a transportation cost coeff, b inventory cost coeff, ICM max inventory cost
    I = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackibility code of an item i
    TI = np.zeros(n) # list of arrival time of each item to plant(days)
    LTI = np.zeros(n) # list of late arrival time of each item (days)
    ETI = np.zeros(n) # list of late early time of each item (days)
    IC = np.zeros(n) # list of inventory costs of each item
    MSC = np.zeros(n)
    p = int(random.uniform(3,n))
    for k in range(n):
        I[k] = random.uniform(1,S)
        MSC[k] = k + p
        ETI[k] = random.uniform(1,E)
        LTI[k] = random.uniform(E,L) #Constraint from special relativity
        TI[k] = random.uniform(ETI[k],LTI[k])
        IC[k] = random.uniform(1,ICM)

    SMSC = np.sort(MSC)
    c = 0
    i = 0
    j = int(SMSC[0])
    while j < n-1: #this block of code makes up for a minimum number of stacks possible
        for k in range(i,j):
            SC[k] = c
        i = j
        j = int(SMSC[j])
        c = c + 1

    d = len(set(SC)) #number of stacks
    VM = [] #stack packing
    VMS = [] #stack inventory cost packing
    S = []
    ST = [] #list of stacks
    ICS = [] #list of stack inventory costs
    for k in range(d):
        S.append(np.where(SC == SC[k]))
        slice1 = [I[S[k][j]] for j in range(len(S[k]))]
        slice2 = [IC[S[k][j]] for j in range(len(S[k]))]
        VM.append(slice1)
        VMS.append(slice2)
        ST.append(np.sum(VM[k]))
        ICS.append(np.sum(VMS[k]))
    SST = np.sort(ST)[::-1]
    B = np.zeros(m) #list of available trucks of length m
    TB = np.zeros(m) #list of arrival time of each truck to plant (days)
    TCt = np.zeros(m) #list of truck costs
    x = np.zeros((m,n)) #Methylated solution (item-truck insertion)
    tabu = np.ones(n) #Methylation list
    for k in range(m):
        B[k] = random.uniform(1,C)
        TB[k] = random.uniform(1,L)
        TCt[k] = random.uniform(1,TC)
        def methylate(n,k):
            f = int(random.uniform(0,n))
            if tabu[f] == 1:
                x[k][f] = 1
                tabu[f] = 0
            else:
                methylate(n,k)
            return x[k]
        for j in range(n//m):
            x[k] = methylate(n,k)
        for j in range(n):
            for k in range(m):
                if x[k][j] == 1:
                    TI[j] = TB[k]
                    break
                else:
                    pass

    def Fitness(y): # y: boolean representation of the solution/chromosome
        A = 0
        for k in range(len(x)):
            A+= b*np.sum(np.sum(IC*(LTI-TI)*x[k])*y)
        return a*np.sum(TCt*y) + A
    #1- Population generation (p,e)
    L = [] # list of K-lists (number of trucks optimization problem)
    TP = [] #Population of solutions
    for k in range(e):
        Y = np.zeros(m)
        for j in range(m):
            Y[j] = int(random.uniform(0,2))
        L.append(Y)
    for j in range(len(L)):
        P = list(itertools.permutations(L[j]))
        TSP = np.zeros((p,m)) #list of p-lists (arrival time problem)
        for k in range(p):
            TSP[k] = P[int(random.uniform(0,len(P)))]
        for l in range(len(TSP)):
            TP.append(TSP[l]) #list of Kp-lists of length ep
    t = 0 #initial generation
    while t < MaxIter:
        #2-Evaluate chromosome fitness:
        Fit = []
        for k in range(len(TP)):
            Fit.append(Fitness(TP[k]))
        #Elitism
        l = Fit.index(np.min(Fit))
        A = Fit
        A.pop(l)
        q = A.index(np.min(A))
        G = A
        G.pop(q)
        u = G.index(np.min(G))
        Elitism = [TP[l],TP[q],TP[u]]
        #3- Natural Selection:
        Selection = []
        Pr = []
        Sum = 0
        for j in range(len(Fit)):
            Sum += Fit[j]
        for k in range(len(Fit)):
            Pr.append(Fit[k]/Sum)
        for m in range(len(Fit)):
            if Pr[m] < random.uniform(0,1):
                Selection.append(TP[m])
        #4- Crossover:
        Crossover = Selection.copy()
        for k in range(len(Crossover)):
            v = int(random.uniform(0,len(Crossover)-1))
            b = int(random.uniform(0,len(Crossover)-1))
            if v != b and np.sum(Crossover[b]) != 0 and np.sum(Crossover[v]) != 0:
                u = int(random.uniform(1,len(Crossover[b])-1))
                X = Crossover[b][u:]
                Crossover[b][u:] = Crossover[v][u:]
                Crossover[v][u:] = X
        #5- Mutation:
        Mutation = Crossover.copy()
        for k in range(len(Mutation)):
            r = int(random.uniform(0,len(Mutation[k])-1))
            if Mutation[k][r] == 1:
                Mutation[k][r] = 0
            else:
                Mutation[k][r] = 1
        TP = Selection.copy()
        for j in range(len(Elitism)):
                TP.append(Elitism[j])
        t = t+1 #next generation
    Sol = []
    for j in range(len(TP)):
        if np.sum(TP[j]) != 0:
            Sol.append(np.sum(TP[j]))

    return Sol, np.min(Sol), np.max(Sol), int(np.mean(Sol)) #Descriptive Statistics of the solution: Sol (list of fittest solutions with their values), its minimum (ideal case) and maximum (worst-case) as well as mean (average-case)

print(BP(50,5,2,2,15,23,1,5,20,20,20,20,30))




