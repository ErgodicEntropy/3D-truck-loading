import numpy as np
import random
import matplotlib.pyplot as plt



n = 100

def pos(p):
    SC = np.zeros(n)
    MSC = np.zeros(n)
    for k in range(n):
        MSC[k] = k + p

    SMSC = np.sort(MSC)
    c = 0
    i = 0
    j = int(SMSC[0])

    while j < n-1 :
        for k in range(i,j):
            SC[k] = c
        c = c + 1
        i = j
        j = int(SMSC[j])
    return [MSC,len(set(SC))]


P = np.zeros(100)
for k in range(100):
    P[k] = pos(k+7)[0][k]

Y = np.zeros(100) #number of stacks
for k in range(100):
    Y[k] = pos(k+7)[1]

plt.plot(P,Y)
plt.xlabel("Minimum Maximum Stackability")
plt.ylabel("Number of stacks")
plt.title("Number of stacks vs  Maximum stackability")
plt.show()


