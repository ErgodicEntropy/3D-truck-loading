#Scalability problem resolution: Adding more constraints

#VM packing problem (Stackable code) + delay optimization: deterministic filtration maximization algorithm (O(nlogn)): non-intrinsic TI (variable) where truck selection based on item's arrival time makes sense, no predefined value and compatible with closed-form explicit fitness function (this doesn't minimize the number of trucks used unless a truck-based approach is adopted)


#Methylation Assumption: To avoid computational redundancy, it's fair to assume that each truck has a pre-defined number of items K in it (to be known).
#Corollary of assumption1: Genetic algorithm will be applied on y[j]s only

#intrinsicality should be matched with elimination otherwise paradox arises
#In this case, TI is just a placeholder variable for TB

#Note: Difference between Plant/Suppler and Plant/Supplier dock is that Plants/Suppliers are akin to brands/companies while their docks (of which there are plausibly many) are the physical headquarters (equivocal)

import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt



def BP(n,m,d,a,b,TC,ICM,ILM,IWM,TMM,TEMM,TMMyM,E,L,S,K,MaxIter): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, TC biggest truck cost, a transportation cost coeff, b inventory cost coeff, ICM max inventory cost.

#DEFINITION OF PARAMETERS, VARIABLES, CONSTRAINTS AND OBJ FUNCTION
    II = [] #Item code
    IM = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackability code of an item i
    MSC = np.zeros(n) #Maximum Stackability code of an item i
    TI = np.zeros(n) # list of arrival time of each item to plant(days)
    LTI = np.zeros(n) # list of late arrival time of each item (days)
    ETI = np.zeros(n) # list of late early time of each item (days)
    IC = np.zeros(n) # list of inventory costs of each item
    IP = np.zeros(n) # list of plants of each item
    IU = np.zeros(n) # list of suppliers of each item
    IG = np.zeros(n) # list of plant docks of each item (TFt dependent)
    IK = np.zeros (n) # list of supplier docks of each item
    IR = np.zeros(n) # list of products of each item
    IL = np.zeros(n) #length of each item
    IW = np.zeros(n) # width of each item
    IH = np.zeros(n) # height of each item
    NH = np.zeros(n) #Nested height of each item
    IO = np.zeros(n) #Forced orientation of each item
    for k in range(n):
        II.append(k)
        IM[k] = random.uniform(1,S)
        MSC[k] = int(random.uniform(d,n-d+1))
        ETI[k] = random.uniform(1,E)
        LTI[k] = random.uniform(E,L) #Constraint from special relativity
        TI[k] = random.uniform(ETI[k],LTI[k])
        IC[k] = random.uniform(1,ICM)
        IL[k] = random.uniform(1,ILM)
        IW[k] = random.uniform(1,IWM)
        IP[k] = int(random.uniform(1,n))
        IU[k] = int(random.uniform(1,n))
        IG[k] = int(random.uniform(1,n))
        IK[k] = int(random.uniform(1,n))
        IR[k] = int(random.uniform(1,n))
        if k <= d:
            SC[k] = k
        else:
            SC[k] = int(random.uniform(1,d+1))
    VM = [] #stack item weights slices packing
    VMS = [] #stack inventory cost packing
    ST = [] #Stack code
    S = [] #S1: stackability code
    U = [] #S1: supplier
    P = [] #S1: plant
    K = [] #S1: supplier dock
    SM = [] #list of stacks weights
    ICS = [] #list of stack inventory costs
    SL = [] #stack length
    SW = [] #stack width
    SH = [] #stack height
    SX0 = [] #initial x coordinate
    SY0 = [] #initial y coordinate
    SZ0 = [] #initial z coordinate
    SXe = [] #extreme x coordinate
    SYe = [] #extreme y coordinate
    SZe = [] #extreme z coordinate
    SOs = [] # Stack forced orientation
    for k in range(len(set(SC))):
        S.append(np.where(SC == SC[k])[0])
        U.append(IU == IU[k])
        P.append(IP == IP[k])
        K.append(IK == IK[k])
    for k in range(len(set(SC))):
        slice1 = [IM[S[k][j]] for j in range(len(S[k]))]
        slice2 = [IC[S[k][j]] for j in range(len(S[k]))]
        sliceS = set([II[S[k][j]] for j in range(len(S[k]))])
        sliceU = set([II[U[k][j]] for j in range(len(U[k]))])
        sliceP = set([II[P[k][j]] for j in range(len(P[k]))])
        sliceK = set([II[K[k][j]] for j in range(len(K[k]))])
        SliceMain = sliceS.intersection(sliceU,sliceP,sliceK) #S1
        VM.append(slice1)
        VMS.append(slice2)
        ICS.append(np.sum(VMS[k]))
        SM.append(np.sum(VM[k]))
        ST.append(SliceMain)
        SX0.append(random.unifom(1,100))
        SY0.append(random.uniform(1,100))
        SZ0.append(0)
    for j in range(len(ST)):
        sl = 0
        sw = 0
        for k in ST[j]:
            sl += IL[k]
            sw += IW[k]
        SL.append(sl)
        SW.append(sw)
    SSM = np.sort(SM)[::-1]
    TT = np.zeros(m) #truck code
    TMtm = np.zeros(m) #maximal weight of stacks (capacity of truck)
    TEMt = np.zeros(m) #maximal stack density
    TMMyt = np.zeros(m) # maximal weight of items above bottom item
    TB = np.zeros(m) #list of arrival time of each truck to plant (days)
    TCt = np.zeros(m) #list of truck costs
    TLt = np.zeros(m) #length of the truck (trailer)
    TWt = np.zeros(m) #width of the truck (trailer)
    THt = np.zeros(m) #height of the truck (trailer)
    TRt = [] # set of candidate products loaded into truck t
    TUt = [] #set of candidate suppliers picked-up by truck t
    TKut = [] # set of candidate supplier docks loaded into truck t
    TGpt = [] # set of candidate plant docks delivered by truck t
    TEt = np.zeros(m) #list of supplier loading order
    TKEut = np.zeros(m) #dock loading order of supplier u of truck t
    TGEpt = np.zeros(m) #dock loading order of plant p of truck t
    TFt = np.zeros(m) #Stack with multiple docks flag
    y = np.zeros(m)
    x = np.zeros((m,d))
    for k in range(m):
        TT[k] = k
        TFt = int(random.uniform(0,2))
        TMtm[k] = random.uniform(1,TMM)
        TEMt[k] = random.uniform(SM/(SL*SW),TEMM) #S7
        TB[k] = random.uniform(1,L)
        TCt[k] = random.uniform(1,TC)#nuances between extra and planned trucks
        y[k] = int(random.uniform(0,2))
        for j in range(int(random.uniform(1,len(IR)))):
            TRt[k].append(IR[j])
        for j in range(int(random.uniform(1,len(IU)))):
            TUt[k].append(IU[j])
        for j in range(int(random.uniform(1,len(IK)))):
            TKut[k].append(IK[j])
        for j in range(int(random.uniform(1,len(IG)))):
            TGpt[k].append(IG[j])
        for j in range(d):
            x[k][j] = int(random.uniform(0,2))
    for l in range(len(TUt)):
        TEt[l] = l
    for l in range(len(TKut)):
        TKEut[l] = l
    for l in range(len(TGpt)):
        TGEpt[l] = l

    for j in range(len(ST)):
        for k in ST[j]:
                IH[k] = random.uniform(1,THt/n)
                NH[k] = random.uniform(1,IH[k])

    for j in range(len(ST)):
        sh = 0
        nh = 0
        for k in ST[j]:
            sh += IH[k]
            nh += NH[k]
        SH.append(math.abs(sh-nh))

    for k in range(m): #S4
        for j in range(d):
            if x[k][j] == 1:
                Item_List_In_Stack = []
                for h in ST[j]:
                    Item_List_In_Stack.append(SM[h])
                BottomItem = Item_List_In_Stack.index(np.max(Item_List_In_Stack))
                MST = ST[j].copy()
                MST.pop(BottomItem)
                SumWeights = 0
                for h in MST[j]:
                    SumWeights += SM[h]
                TMMyt[k] = random.uniform(SumWeights,TMMyM)
            else:
                pass

    for k in range(m):
        if TFt[k] == 0: #S2
            for j in range(d):
                if x[k][j] == 1:
                    K = int(random.uniform(1,n))
                    for f in ST[j]:
                        IG[f] = K
                else:
                    pass
        else: #S3
            for j in range(d):
                if x[k][j] == 1:
                    W = int(random.uniform(1,n))
                    for f in ST[j]:
                        r = int(random.uniform(0,2))
                        if r == 1:
                            IG[f] = W
                        else:
                            IG[f] = W+1
                else:
                    pass

    for k in range(m): #P1
        for j in range(d):
            if x[k][j] == 1:
                SXe.append(random.uniform(SX0,TLt))
                SYe.append(random.uniform(SY0,TWt))
                SZe.append(SH[j]+SZ0[j])
            else:
                pass

    L = [int(random.uniform(0,2)) for j in range(d)] #Lenghtwise list
    W = np.ones(d) #Widthwise list
    for j in range(d):
        if L[j] == 1:
            W[j] = 0

    def Rotation(j):
        if L[j] == 1:
            L[j] = 0
            W[j] = 1
        if W[j] == 1:
            W[j] = 0
            L[j] = 1

    for j in range(d):
        if L[j] == 1:
            SXe[j] = SL[j] + SX0[j]
            SYe[j] = SW[j] + SY0[j]
        if W[j] == 1:
            SXe[j] = SW[j] + SX0[j]
            SYe[j] = SL[j] + SY0[j]

    for j in range(d): #S4
        f = random.choice(ST[j])
        SOs[j] = IO[f]
















