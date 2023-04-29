#Tabu Search: a memory-based local optimization algorithm that makes the use of a tabu list of temporarily (depends on the tabu tenure/expiration rate) prohibited solutions either because they were visted previously or violate a certain user-defined rule, in order to avoid premature convergence to local optima and explore the search space. In Tabu Search, there is a trade-off between memory requirement and exploration process; it is favorable to maintain a balance between the two through an intermediate-length tabu list


#Methylation Hypothesis: To avoid computational redundancy, it's fair to assume that each truck has a pre-defined number of items K = n/m in it (to be known). This hypothesis treats the item-truck insertion operation as a black box aposteriori (meaning it already happened without knowing how)
#Corollary of this assumption: Tabu Search will be applied on y[j]s only


import numpy as np
import math
import itertools
import functools
import random
import matplotlib.pyplot as plt




def MBP(n,m,a,b,TC,ICM,E,L,S,C,MaxIter): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, TC biggest truck cost, a transportation cost coeff, b inventory cost coeff, ICM max inventory cost
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
    tabux = np.ones(n) #Methylation list
    for k in range(m):
        B[k] = random.uniform(1,C)
        TB[k] = random.uniform(1,L)
        TCt[k] = random.uniform(1,TC)
        def methylate(n,k):
            f = int(random.uniform(0,n))
            if tabux[f] == 1:
                x[k][f] = 1
                tabux[f] = 0
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

    def SolGen(m):
        y = np.zeros(m)
        for k in range(m):
            y[k] = int(random.uniform(0,2))
        return y

    def SolCompare(X,Y):
        M = list(map(lambda a,b: a == b,X,Y))
        return np.all(M) #functools.reduce(lambda x,y: x and y, map(lambda p,q: p == q,X,Y),True) is also valid via functools library


    def Fitness(y):
        A = 0
        for k in range(len(x)):
            A+= b*np.sum(np.sum(IC*(LTI-TI)*x[k])*y)
        return a*np.sum(TCt*y) + A

    q = 10 #tabu tenure (1/expiration rate)



    #Implementation
    tabu = np.zeros((q,m))
    for j in range(q):
        tabu[j] = SolGen(m)

    def SolCheck(solution):
        BolF = True
        for r in range(q):
            BolF*= SolCompare(solution,tabu[r])
        return BolF

    y0 = SolGen(m) #initial solution

    w = 0
    while w < MaxIter:
        sol = y0.copy()
        v = int(random.uniform(0,q+1)) #Changing one element at a time ( step size=1 )
        if sol[v] == 1:
            sol[v] = 0
        else:
            sol[v] = 1 #The neighborhood topology is defined by the sum(XOR operator) distance measure/norm between booelan arrays

        ΔF = Fitness(sol) - Fitness(y0)
        if ΔF <= 0 and SolCheck(sol) == True:
            y0 = sol
            tabu.pop(0)
            tabu.append(sol)
        elif SolCheck(sol) == True:
            y0 = sol
            tabu.pop(0)
            tabu.append(sol)
        w = w + 1

    L = np.zeros(q)
    for k in range(q):
        L[k]= Fitness(tabu[k])

    B = int(np.where(L == np.min(L))[0])

    return np.sum(tabu[B]) #Aspiration criterion (otherwise return the last element inserted in the tabu list)




print(MBP(100,50,2,2,18,20,1,5,20,200,100)) #Methylated TS



def NMBP(n,m,a,b,TC,ICM,E,L,S,C,MaxIter): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, TC biggest truck cost, a transportation cost coeff, b inventory cost coeff, ICM max inventory cost
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
    for k in range(m):
        B[k] = random.uniform(1,C)
        TB[k] = random.uniform(1,L)
        TCt[k] = random.uniform(1,TC)

    def SolGen(n,m):
        y = np.zeros(m)
        x = np.zeros((m,n))
        for k in range(m):
            y[k] = int(random.uniform(0,2))
            for j in range(n):
                x[k][j] = int(random.uniform(0,2))
        return [x,y]

    def SolCompareD1(X,Y): #compare lists with dimension 1
        M = list(map(lambda a,b: a == b,X,Y))
        return np.all(M) #functools.reduce(lambda x,y: x and y, map(lambda p,q: p == q,X,Y),True) is also valid via functools library

    def SolCompareD2(X,Y): #compare lists with dimension 2
        Bol = True
        for j in range(len(X)): #or len(Y) as we are ignoring the other case naturally false
            Bol*= SolCompareD1(X[j],Y[j])
        return Bol


    def Fitness(x,y):
        A = 0
        for k in range(len(x)):
            A+= b*np.sum(np.sum(IC*(LTI-TI)*x[k])*y)
        return a*np.sum(TCt*y) + A

    q = 10 #tabu tenure (1/expiration rate)



    #Implementation
    tabuX = np.zeros((q,m,n))
    for j in range(q):
        tabuX[j] = SolGen(n,m)[0]

    tabuY = np.zeros((q,m))
    for j in range(q):
        tabuY[j] = SolGen(n,m)[1]


    def BiSolCheck(solutionX,solutionY):#This function can be separated into two check funcs
        BolFX = True
        BolFY = True
        for r in range(q):
            BolFX*= SolCompareD2(solutionX,tabuX[r])
            BolFY*= SolCompareD1(solutionY,tabuY[r])

        return BolFX*BolFY


    x0 = SolGen(n,m)[0] #initial solution (insertion)
    y0 = SolGen(n,m)[1] #initial solution (usage)

    w = 0
    while w < MaxIter:
        solX = x0.copy()
        solY = y0.copy()
        v = int(random.uniform(0,m)) #Changing one element at a time ( step size=1 )
        b = int(random.uniform(0,n))
        if solX[v][b] == 1:
            solX[v][b] = 0
        else:
            solX[v][b] = 1

        if solY[v] == 1:
            solY[v] = 0
        else:
            solY[v] = 1 #The neighborhood topology is defined by the sum(XOR operator) distance measure/norm between booelan arrays


        ΔF = Fitness(solX,solY) - Fitness(x0,y0)
        if ΔF <= 0 and BiSolCheck(solX,solY) == True:
            x0 = solX
            y0 = solY
            tabuX.pop(0)
            tabuX.append(solX)
            tabuY.pop(0)
            tabuY.append(sol)
        elif BiSolCheck(solX,solY) == True:
            x0 = solX
            y0 = solY
            tabuX.pop(0)
            tabuX.append(solX)
            tabuY.pop(0)
            tabuY.append(solY)
        w = w + 1

    randomY = SolGen(n,m)[1]#Optimizing up to an arbitrary random y value
    LX = np.zeros(q)
    for k in range(q):
        LX[k]= Fitness(tabuX[k],randomY)

    randomX = SolGen(n,n)[0]#Optimizing up to an arbitrary random x value
    LY = np.zeros(q)
    for k in range(q):
        LY[k]= Fitness(randomX,tabuY[k])


    BX = int(np.where(LX == np.min(LX))[0])
    BY = int(np.where(LY == np.min(LY))[0])


    return tabuX[BX], np.sum(tabuY[BY]) #Aspiration criterion (otherwise return the last element inserted in the tabu list)




print(NMBP(100,50,2,2,18,20,1,5,20,200,100)) #Non-Methylated TS









