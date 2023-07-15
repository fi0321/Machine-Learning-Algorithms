import numpy as np
import math
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt


with open("targets1.txt") as textFile:
    lines=[line.split() for line in textFile]

with open("attributes1.txt") as textFile:
    lines1=[line.split() for line in textFile]

lines_=np.array(lines)
t=np.array(lines).astype(np.float)
X=np.loadtxt("attributes1.txt")

w2=np.random.rand(4,1)*0.1-0.05

X=np.insert(X,3,-1, axis=1)


def Recall(x, w):
    activation=0
    g=0
    for i in range(len(x)):
        activation=activation+w[i,:]*x[i]

    if (activation>0):
        g=1
    else:
        g=0
    return g
def Train(X,t,eta,iter):
    w=np.random.rand(4,1)*0.1-0.05
    m=np.loadtxt("attributes1.txt")
    m=np.insert(m,3,-1, axis=1)
    print("Initial weight matrix is ", w)
    for i in range(iter):
        outputlist=np.matrix("0;0;0;0")
        for k in range(len(X)):
            outputlist[k]=Recall(X[k],w)


        for j in range(len(X)):

            if (outputlist[j]!=t[j]):
                for p in range(len(X[j])):
                    w[p]=w[p]+(eta*(t[j]-outputlist[j]))*X[j][p]
        print("Weight matrix after update round ",i+1,": ",w)    
        print("output pattern: ",outputlist)
                
    return w


w1=Train(X,t,0.2,12)

