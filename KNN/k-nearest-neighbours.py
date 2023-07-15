import numpy as np
import math
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
with open("targets.txt") as textFile:
    lines=[line.split() for line in textFile]

lines_=np.array(lines)



m=np.loadtxt("attributes.txt")
c=np.loadtxt("targets.txt")
#print(lines_)
def CrossVal10(m, c, k):
    size = np.size(m, axis=0) #objects count
    fsize = size//10   #size of each fold
    errorList=[None]*fsize
    summingerror=0
    avgError=0
    for i in range(10):
        trainx = np.delete(m, range(i*fsize, (i+1)*fsize), axis=0)
        trainl = np.delete(c, range(i*fsize, (i+1)*fsize))
        testx = m[i*fsize:(i+1)*fsize,:]
        testl = c[i*fsize:(i+1)*fsize]
        testl1=KNN(trainx, trainl, testx, k)
        each_Error=0
        for j in range(len(testl1)):
            if(testl1[j]!=testl[j]):
                each_Error=each_Error+1
        each_Error=each_Error/(len(testl))
        errorList[i]=each_Error
    for p in range(len(errorList)):
        summingerror=summingerror+errorList[p]
    avgError=summingerror/(len(errorList))
    return avgError


        
    
def KNN(trainx, trainl, testx, k):
    all_Distances=[]
    size = np.size(testx, axis=0) #objects count
    testl1=[None]*len(testx)
    for i in range(size):
        m = np.array([testx[i,:]])  
        dist=np.sum((trainx-np.repeat(m,np.size(trainx,axis=0), axis=0))**2, axis=1)
        all_Distances.append(dist)
    len_distances=len(all_Distances)
    for j in range(len_distances):
        
        ind=np.argsort(all_Distances[j])
        kclasses=[]
        for l in range(k):
            kclasses.append(trainl[ind[l]])
        counterZero=0
        counterOne=0
        for y in range(len(kclasses)):
            if (kclasses[y]==0):
                counterZero=counterZero+1
            else:
                counterOne=counterOne+1
        if(counterZero>counterOne):
            testl1[j]=0
        elif(counterZero==counterOne):
            testl1[j]=1
        else:
            testl1[j]=1



    return testl1

def lowesterror(m,c):
    AllErrors=[None]*30
    for i in range(30):
        AllErrors[i]=CrossVal10(m,c,i+1)
    lowesterror=AllErrors.index(min(AllErrors))
    return lowesterror+1


# for i in range(30):
#    average=CrossVal10(m,c,i+1)
#    print ("for k= ", i+1, "the values is", average)

lowestk=lowesterror(m,c)

def plotit(m,c) :
    k=[None]*30
    misclass=[None]*30

    for i in range (30) :
        k[i]=i+1
        misclass[i]=CrossVal10(m,c,i+1)*100

    xint = range(min(k), math.ceil(max(k))+1)
    plt.figure(figsize=(14,6))
    plt.xticks(xint)

    plt.plot(k,misclass,'ro--')


    plt.xlabel('Value of K')
    plt.ylabel('CV error percentage in %')
    plt.show()

print("The best K is",lowestk, "with CV error=", CrossVal10(m,c,lowestk), "or in percentage ", CrossVal10(m,c,lowestk)*100, "%")

plotit(m,c)
