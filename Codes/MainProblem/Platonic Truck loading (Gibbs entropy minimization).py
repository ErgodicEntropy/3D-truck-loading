#VM packing problem (Gibbs entropy minimization): Space-Scalability problem has to be solved first: 1- Direct: python tool that deals with geometric spaces 2- Platonic representations of data structures (rank-3 Tensor for 3D problem )

#K-Complex MetaOpt:Our problems is divided into 3 fundamental subproblems that are interconnected between them because the problem proposed is complex in the sense of emergence and non-linearity between the subproblems (min function is non-linear) that exhibit feedback loops. Our first subproblem is the truck usage (K=1), second subproblem is item/stack/truck insertion (K=2) and third subproblem is minimization of Gibbs entropy for an optimum geometric placement of items in the truck (its dual is the maximization of compacity/packing density of the items; eliminating the unnecessary void between items).\\
#Complexity arises from the fact that subproblem (K=1) is the dual optimization problem of the subprobem (K=2) (Recovering vs Packing)

#Gibbs entropy is defined as the amount of disorder in the system (naturally high corresponding to random sampling) thus order-specific and unequally distributed (some states are more favored than others; second law of thermodynamics). In our case, Gibbs entropy is the amount of weight-unconstrained void (thus unnecessary) between stacks (void is actually the cause of high entropy to be precise). This analysis assumes that the loading constraints are met but it tries to maximize loading possibilies ie; number of potential future items/stacks to be loaded (primal problem of entropy dual problem): technically "potential" loading problem K=3: pre-defined loading problem (K =1,2) -> post-loading problem (K = 3) = maximization of abs(card(pre-SIs)-card(post-SIs)) : Truck-based approach GS Gibbs entropy (dual),G geometry/packing density/compacity (primal)

#Minimization of Gibbs entropy = Maximization of packing density/compacity = Elimination of collective void between every stack (it's emergent): Truck-based approach

#Gibbs entropy is the cause, packing density is the effect (forward problems vs inverse problems in numerical analysis)

#Minimizing Gibbs entropy is equivalent to finding a maximum independent set in a graph represented by intersection sets (Dual Recovering problem )

#Redundant void = 1 - packing density

#Packing density depends on the organization of the stacks in the truck and the  truck maximum authorized loading weight

#METHODOLOGY 1: Geometry-specific

#Method 1: Exhaustive search using enumerative combinatorics:

#Combinatorial/Conformational entropy is the number of ways of rearranging a system (a molecule or any thermodynamic system). Combinatorial entropy is used to enumerate all exhaustive possibilities of 5 degrees of freedom: translation sx,sy,sz (continuously infinite, discretly finite) , Horizontal rotation, Forced Orientation SOs (4->2 because of structural symmetry) : lenghtwise, widthwise =>indiviudal void -> recursion -> collective void

#Method 2: low-entropy virtual Placeholder Meshing stack fitting (error estimation) using number theory



#METHODOLOGY 2: Platonic representation via correlation-based model

#Graph-based method: working out the inverse problem! (from effects to causes): minimizing void or maximizing packing density because there is no explicit formula for geometry-specific Gibbs entropy. Independent sets can help with that because finding maximal independent sets is platonically equivalent to maximizing packing density/minimizing redundant void (under the percolation threshold)

#Each truck has its own independence number that is positively correlated with its size and optimal arrangement (Platonic representation via independent sets)

#Correlation-based Modeling:  Trucks (truck size = ceiling efficiency of optimal arrangement) vs Graphs (Sparsity ratio |V|/|E| = number of zeros in the AdjMat (Weak, Immediate, Strong conditions) resulting from topological shape (edge-dependent) of the graph ~ 1/occupation probability) -> high Sparsity ratio = Occupation probability (edge-dependent) far from percolation thershold: Prob(MaxIndSet with high IN) = 1 - OP

#NOTE: Maximum graph degree is also a measure of its independence number

#There is a tradeoff between the difficulty of finding an independent set and the quality of the independent set (independent number): rock vs gold analogy

#Truck size of a truck ~ Max(Sparsity ratio) of a graph (efficiency ceiling)
#Arrangement ~ reducing |E| of the isomorphic graph to reach the Max(SR) guaranteed by optimal arrangement

#Case 1: |V| is concretely the number of stacks (unlike |E| being abstract property of the isomorphic graph to a given truck): there is a dependence problem between different truck loadings (first truck is always favored because it has the highest |V| that decreases recursively therefore taking from other trucks sparsity ratios thus decreasing their independence number regardless of optimal arragement). Albeit that being the case, there is a "good enough" solution that prioritize trucks with the highest truck size (just like in F-F decreasing algo)

#Case 2: |V| is abstract just like |E|: in this case there is no dependence problem between different truck loadings. Arrangement in this case will manipulate both |E| and |V| up to the Max(SR) dictated by truck size. This sounds good because one wouldn't like the concrete to disrupt the isomorphism between trucks and graphs (because trucks are ontologically independent from stacks). However, this is discouraged because the whole point of maximal independent set problem hinges on the fact that |V| is concretely the number of stacks

#From case 1 and 2: there is a tradeoff between truck loading independence and Independent Set problem via intersection sets graph representation validity

#Algorithm/Resolution:







import numpy as np
import math
import itertools
import random
import matplotlib.pyplot as plt


def BP(n,m,TC,ICM,TMM,TEMM,TMMyM,E,L,S): #n is the number of items, m is the total number of available bins, S maximum size of items, C maximum capacity of bins, d number of stacks formed (d < n), E earliest time, L latest time, TC biggest truck cost, a transportation cost coeff, b inventory cost coeff, ICM max inventory cost.


#DEFINITION OF PARAMETERS, VARIABLES, CONSTRAINTS AND OBJ FUNCTION
    II = [] #Item code
    IM = np.zeros(n) #list of items (indices) and their sizes (list values)
    SC = np.zeros(n) #Stackability code of an item i
    MSC = np.zeros(n) #Maximum Stackability code of an item i
    LTI = np.zeros(n) # list of late arrival time of each item (days)
    ETI = np.zeros(n) # list of late early time of each item (days)
    IC = np.zeros(n) # list of inventory costs of each item
    IP = np.zeros(n) # list of plants of each item
    IU = np.zeros(n) # list of suppliers of each item
    IG = np.zeros(n) # list of plant docks of each item (TFt dependent)
    IK = np.zeros (n) # list of supplier docks of each item
    IR = np.zeros(n) # list of products of each item
    p = int(random.uniform(3,n))
    for k in range(n):
        II.append(k)
        IM[k] = random.uniform(1,S)
        MSC[k] = k + p
        ETI[k] = random.uniform(1,E)
        LTI[k] = random.uniform(E,L) #Constraint from special relativity
        IC[k] = random.uniform(1,ICM)
        IP[k] = int(random.uniform(1,n))
        IU[k] = int(random.uniform(1,n))
        IG[k] = int(random.uniform(1,n))
        IK[k] = int(random.uniform(1,n))
        IR[k] = int(random.uniform(1,n))

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
    VM = [] #stack item weights slices packing
    VMS = [] #stack inventory cost packing
    ST = [] #Stack code
    S = [] #S1: stackability code
    U = [] #S1: supplier
    P = [] #S1: plant
    K = [] #S1: supplier dock
    SM = [] #list of stacks weights
    ICS = [] #list of stack inventory costs
    for k in range(d):
        S.append(np.where(SC == SC[k])[0])
        U.append(IU == IU[k])
        P.append(IP == IP[k])
        K.append(IK == IK[k])
        slice1 = [IM[S[k][j]] for j in range(len(S[k]))]
        slice2 = [IC[S[k][j]] for j in range(len(S[k]))]
        sliceS = set([II[S[k][j]] for j in range(len(S[k]))])
        sliceU = set([II[S[k][j]] for j in range(len(S[k]))])
        sliceP = set([II[S[k][j]] for j in range(len(S[k]))])
        sliceK = set([II[S[k][j]] for j in range(len(S[k]))])
        SliceMain = sliceS.intersection(sliceU,sliceP,sliceK) #S1
        VM.append(slice1)
        VMS.append(slice2)
        ICS.append(np.sum(VMS[k]))
        SM.append(np.sum(VM[k]))
        ST.append(SliceMain)


    TT = np.zeros(m) #truck code
    TMtm = np.zeros(m) #maximal weight of stacks (capacity of truck)
    TEMt = np.zeros(m) #maximal stack density
    TMMyt = np.zeros(m) # maximal weight of items above bottom item
    TB = np.zeros(m) #list of arrival time of each truck to plant (days)
    TCt = np.zeros(m) #list of truck costs
    TRt = [] # set of candidate products loaded into truck t
    TUt = [] #set of candidate suppliers picked-up by truck t
    TKut = [] # set of candidate supplier docks loaded into truck t
    TGpt = [] # set of candidate plant docks delivered by truck t
    TEt = np.zeros(m) #list of supplier loading order
    TKEut = np.zeros(m) #dock loading order of supplier u of truck t
    TGEpt = np.zeros(m) #dock loading order of plant p of truck t
    TFt = np.zeros(m) #Stack with multiple docks flag
    for k in range(m):
        TT[k] = k
        TFt = int(random.uniform(0,2))
        TMtm[k] = random.uniform(1,TMM)
        TEMt[k] = random.uniform(1,TEMM)
        TMMyt[k] = random.uniform(1,TMMyM)
        TB[k] = random.uniform(1,L)
        TCt[k] = random.uniform(1,TC)#nuances between extra and planned trucks





    def Vertex_Degree(v,G):
        return np.sum(G[v])
    def Secondary_Vertex_Degree(v,G):
        s = 0
        Adj = np.where(G[v] == 1)[0]
        for k in range(len(Adj)):
            s+= Vertex_Degree(Adj[k],G)
        return s

    def count_zeros(G):
        X = len(G)**2 - np.count_nonzero(G) # |V|/|E| Sparsity ratio (replacable by Δ(G))
        return X #Max = len(G)**2, Min = 0

    def GMD(G): #Maximum degree of a graph (proxy for sparsity ratio)
        L = []
        for k in range(len(G)):
            L.append(Vertex_Degree(k,G))
        Δ = np.max(L)
        I = int(np.where(L == Δ)[0][0])
        return [Δ,I] #Max = len(G) - 1, Min = 1


    def Iso_Graph(r,e): #Platonic isomorphism between trucks and graphs
        if r <= 0:
            return []
        G = np.zeros((r,r))
        for k in range(r):
            for j in range(r):
                if k != j:
                    G[j][k] = int(random.uniform(0,2))
                    G[k][j] = G[j][k]
        D = GMD(G)[0] # max(GMD) = r - 1
        while D > math.floor(1/TMtm[e]): #TMtm = Max(|V|/|E|) = r**2 or Min(GMD) = 1
            p = GMD(G)[1]
            t = int(random.uniform(0,len(G[p])))
            if G[p][t] == 1:
                G[p][t] = 0 #Platonic Arrangement (reducing |E| by 1 an iteration)
            D = GMD(G)[0]
        return G



    def MIS(G): #Improved Greedy algorithm (favoring least degree vertices)
        MIS = []
        DegList = []
        SecDegList = []
        for k in range(len(G)):
            DegList.append(Vertex_Degree(k,G))
            SecDegList.append(Secondary_Vertex_Degree(k,G))
        DegArray = np.array(DegList)
        SecDegArray = np.array(SecDegList)
        SDegArray = np.sort(DegArray)
        h = 0
        tabu = np.ones(len(G))
        while count_zeros(G) <= len(G)**2 and h < len(G):
            if np.sum(G[h]) != 0:
                o = np.where(DegArray == SDegArray[h])[0]
            else:
                pass
            SO = []
            if len(o) > 1:
                for k in o:
                    SO.append(SecDegArray[k])
                v = random.choice(np.where(SecDegArray == np.max(SO))[0])
            else:
                v = int(o)
            if tabu[v] == 1:
                Neighbors = np.where(G[v] == 1)[0]
                for j in Neighbors:
                    for k in range(len(G[j])):
                        G[j][k] = 0
                        G[k][j] = 0
                        G[v][k] = 0
                        G[k][v] = 0
                MIS.append(v)
                tabu[v] = 0
            h = h + 1

        return MIS

    def Constraint(MISt,l):
        WeightSum = 0
        StackArrivalTimeDiff = []
        StackMinIndex = []
        for k in MISt:
            WeightSum += SM[k]
            ItemArrivalTimeDiff = []
            ItemIndex = []
            for j in ST[k]:
                ItemArrivalTimeDiff.append(LTI[j]-ETI[j])
                ItemIndex.append(j)
            MIN = np.min(ItemArrivalTimeDiff)
            StackArrivalTimeDiff.append(MIN)
            StackMinIndex.append(ItemIndex[random.choice(np.where(np.array(ItemArrivalTimeDiff) == MIN)[0])])
        MINS = np.min(StackArrivalTimeDiff)
        q = StackMinIndex[random.choice(np.where(np.array(StackArrivalTimeDiff)==MINS)[0])]
        return bool(TB[l] <= LTI[q] or TB[l] >= ETI[q] or np.sum(WeightSum) <= TMtm[l])



#RESOLUTION:

    TBM = TB + TMtm
    TBMM = np.sort(TBM)[::-1] #Truck usage K = 1
    e = 0
    j = int(np.where(TBM == TBMM[e])[0]) #favoring latest arrival times
    Gt = Iso_Graph(d,j)
    while len(Gt) > 0 and e <= len(TBMM):
        MISt = MIS(Gt)
        ConstrainedMIS = []
        for k in range(20):
            while Constraint(MISt,j) == False:
                MISt = MIS(Gt)
            ConstrainedMIS.append(len(MISt))
        GS = np.max(ConstrainedMIS) #Gone Stacks
        e = e + 1
        j = int(np.where(TBM == TBMM[e])[0])
        Gt = Iso_Graph(d-GS,j)
    return e + 1



print(BP(100,500,18,20,25,200,200,1,5,20))

#https://en.wikipedia.org/wiki/Packing_problems
#https://en.wikipedia.org/wiki/Rectangle_packing
#https://en.wikipedia.org/wiki/Sphere_packing
# https://github.com/Pithikos/python-rectangles
# https://github.com/MeshInspector/MeshLib
# https://code.google.com/archive/p/pyeuclid/source
# https://docs.scipy.org/doc/scipy/reference/spatial.html
# https://pypi.org/project/rectangle-packing-solver/
















































