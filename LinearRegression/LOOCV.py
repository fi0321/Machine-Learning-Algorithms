import numpy as np
import matplotlib.pyplot as plt

with open("synthdata.txt") as textFile:
    lines=[line.split() for line in textFile]

lines_=np.array(lines)
#task a
def polynomial(n,f) :
    att = f[:,0].astype(np.float)[np.newaxis].T
    t = f[:,1].astype(np.float)[np.newaxis].T
    X = np.ones((len(t),1))
    for i in range(n) :
        X = np.hstack((X,att**(i+1)))
        v = np.linalg.inv(X.T@X)
        w = v@X.T@t
    return w         

def Xw(x,w) :
    z = np.ones(len(w))
    for i in range(len(w)):
        z[i] = x**i
    return z@w
    
#Task b
def loocv(n,f):
    sum = 0
    for i in  range( len(f)):
        f_mask = f
        f_mask =  np.delete(f_mask, i, 0)                         
        att = f_mask[:,0].astype(np.float)[np.newaxis].T
        t = f[:,1].astype(np.float)[np.newaxis].T
        w = polynomial(n,f_mask)
        f_i = f[i].astype(np.float)
        xw = Xw(f_i[0],w)
        sum = sum+(xw-f_i[1])**2
        
    return sum/len(f)

#Task c

def plotit(n,f) :
    Loss = []
    P = []
    for i in range (1,n+1) :
        Loss.append(loocv(i,f))
        P.append(i)
    plt.plot(P,Loss)
    plt.xlabel('Polynomial order')
    plt.ylabel('Mean LOOCV Loss')
    plt.show()
def param(n,f) :
    Loss = []
    P = []
    for i in range (1,n+1) :
        Loss.append(loocv(i,f))
        P.append(i)
    print(Loss)
    ml = min(Loss)
    o = Loss.index(ml)+1
    for i in  range( len(f)):
        f_mask = f
        f_mask =  np.delete(f_mask, i, 0)                         
        att = f_mask[:,0].astype(np.float)[np.newaxis].T
        t = f[:,1].astype(np.float)[np.newaxis].T
        w = polynomial(o,f_mask)

    
    return ml,o,w
ml,o,w = param(8,lines_)


print('For the smallest LOOCV ')
print('Mean Loss')
print(ml)
print('Order')
print(o)
print('Coefficients')
print(w)
plotit(8,lines_)


Loss = []
P = []
for i in range (1,6) :
    Loss.append(loocv(i,lines_))
    P.append(i)
print("this is loss", Loss)
print("this is P", P)