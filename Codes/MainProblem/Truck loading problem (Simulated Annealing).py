#Simulated annealing: this global optimization memoryless algorithm is inspired from solid mechanics process of annealing solids; heating up solids up to a limit point to increase their ductility and obtain regular crystals (Enzyme-Catalyzer activation energy lowering analogy is also valid). Ductility is the degree of tolerance/acceptance of bad solutions for search space exploration. In SA, heating is equivalent to being generous while cooling (it should necessarily be progressive in  order to avoid premature convergence) is equivalent to being greedy/exploitative = Balancing the exploration-exploitation tradeoff.

#Simulated Annealing works relatively perfectly on functions with "Weierstrassian" topology (ie; functions with stochastically many local optima). Examples: Ackley function, eggholder function, Weierstrass function (obviously)...etc


#Methylation Hypothesis: To avoid computational redundancy, it's fair to assume that each truck has a pre-defined number of items K = n/m in it (to be known). This hypothesis treats the item-truck insertion operation as a black box aposteriori (meaning it already happened without knowing how)
#Corollary of this assumption: Simulated annealing will be applied on y[j]s only


import numpy as np
import math
import itertools
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

    def Fitness(y):
        A = 0
        for k in range(len(x)):
            A+= b*np.sum(np.sum(IC*(LTI-TI)*x[k])*y)
        return a*np.sum(TCt*y) + A

    T0 = 4 #initial temperature
    a = 0.8 #cooling factor
    TL = 100 #Temperature length
    y0 = SolGen(m) #initial solution

    #Implementation1
    t = 0
    while t < MaxIter:
        for j in range(TL):
            sol = SolGen(m) #randomized solutions
            ΔF = Fitness(sol) - Fitness(y0)
            if ΔF <= 0:
                y0 = sol
            else:
                r = int(random.uniform(0,2))
                if math.exp(-ΔF/T0) >= r:
                    y0 = sol

        t = t + 1
        T0 = a*T0

    return np.sum(y0)

#this SA algorithm can be improved on the level of randomization/exploring the search space by adding a tabu list to it that store the randomized solutions went over previously as to remove redundancy and avoid computational waste (intermediate tabu length is favored): Tabu-driven Simulated Annealing


    #Implementation2

    tabu = np.zeros((MaxIter,TL,m)).tolist()
    def RecurseGen(k,g,h):
        BolF = True
        sol = SolGen(k) #randomized solutions
        for r in range(g):
            for o in range(h):
                BolF*= bool(sol != tabu[r][o])
        if BolF == True:
            tabu[g][h].append(sol)
        else:
            RecurseGen(k,g,h)
        return sol

    w = 0
    while w < MaxIter:
        for j in range(TL):
            sol = RecurseGen(m,w,j)
            ΔF = Fitness(sol) - Fitness(y0)
            if ΔF <= 0:
                y0 = sol
            else:
                r = int(random.uniform(0,2))
                if math.exp(-ΔF/T0) >= r:
                    y0 = sol
        w = w + 1
        T0 = a*T0

    return np.sum(y0)

#For implementation 2, it gives a deprecation warning alluding that list-to-list comparison functionality/feature is a bit outdated and might not work for future versinos of Python. However, that's not a big deal and definitely not something to worry about (the results are printed nonetheless). If one scales down the comparison to elementwise level, a maximum recursion depth problem pops up

print(MBP(100,50,2,2,18,20,1,5,20,200,100)) #Methylated SA





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
        x = np.zeros((m,n))
        y = np.zeros(m)
        for k in range(m):
            y[k] = int(random.uniform(0,2))
            for j in range(n):
                x[k][j] = int(random.uniform(0,2))
        return [x,y]

    def Fitness(x,y):
        A = 0
        for k in range(len(x)):
            A+= b*np.sum(np.sum(IC*(LTI-TI)*x[k])*y)
        return a*np.sum(TCt*y) + A

    T0 = 4 #initial temperature
    a = 0.8 #cooling factor
    TL = 100 #Temperature length
    x0 = SolGen(n,m)[0] #initial solution (insertion)
    y0 = SolGen(n,m)[1] #intiial solution (usage)

    #Implementation
    t = 0
    while t < MaxIter:
        for j in range(TL):
            solX = SolGen(n,m)[0] #randomized solutions
            solY = SolGen(n,m)[1] #randomized solutions
            ΔF = Fitness(solX,solY) - Fitness(x0,y0)
            if ΔF <= 0: #Collective assigment (there are others)
                x0 = solX
                y0 = solY
            else:
                r = int(random.uniform(0,2))
                if math.exp(-ΔF/T0) >= r:
                    x0 = solX
                    y0 = solY

        t = t + 1
        T0 = a*T0

    return np.sum(y0)


print(NMBP(100,50,2,2,18,20,1,5,20,200,100)) #Non-Methylated SA




