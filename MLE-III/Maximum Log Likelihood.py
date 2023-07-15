import numpy as np
import matplotlib.pyplot as plt

with open("olympic_men.txt") as textFile:
    lines=[line.split() for line in textFile]

lines_=np.array(lines)



def polynomial(n,f) :
    att = f[:,0].astype(np.float)
    min_att=min(att)
    att=att[np.newaxis].T
    t = f[:,1].astype(np.float)[np.newaxis].T
    X = np.ones((len(t),1))
    for i in range(n) :
        X = np.hstack((X,(att-min_att)**(i+1)))
        v = np.linalg.inv(X.T@X)
        w = v@X.T@t
    return w , X ,t       



def variance(n,f):
    w , X , t= polynomial(n,f)
    l=X@w
    b=t-l
    a=np.transpose(b)
    q=(1/(len(X)))*(a@b)
    return q


#Task c
def covariance(n,f):
    w , X , t= polynomial(n,f)
    g=len(X)
    o=0
    for i in range(g):
        c=(X[i,:]@w)
        d=(t[i,0]-c)**2
        o=o+d
    lr=np.sqrt((1/g)*o)  
    return lr


def logLikelihood(n,f):
    w , X , t= polynomial(n,f)
    sigma_sq=variance(n,f)
    sigma=covariance(n,f)
    g=len(X)
    o=0
    for i in range(g):
        c=(X[i,:]@w)
        d=(t[i,0]-c)**2
        o=o+d
    th=(1/(2*sigma_sq))
    third=th*o
    second=g*(np.log(sigma))
    first=(g/2)*(np.log(2*(np.pi)))
    whole=-first-second-third
    return whole

def plotit(n,f) :
    loglike = []
    variance2 = []
    order=[]
    for i in range (1,n+1) :
        loglike.append(logLikelihood(i,f))
        variance2.append(variance(i,f))
        order.append(i)
   
    print(loglike)
    plt.plot(order,loglike,'o', order, variance2)
    plt.xlabel('Polynomial order')
    plt.ylabel('Variance/Maximum loglikelihood')
    plt.show()

plotit(5,lines_)

print("this is loglike", loglike)
print("this is variance", variance2)
print("this is order", order)
print("first", logLikelihood(1,lines_))

sigma1=covariance(1,lines_)
whole1=logLikelihood(5,lines_)
k=min(lines_[:,0].astype(np.float).T)
w,X,t=polynomial(1, lines_)
print(sigma1)
print("this is whole", whole1)

