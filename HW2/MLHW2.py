# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:41:36 2018

@author: Susan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# making phi matrix
def base_func(x,j,M,s):
    muj = np.asarray(4*j/M).reshape(1,len(j))   # 各種不同muj
    phi = (np.tile(x,M)-np.tile(muj,(x.shape[0],1)))/s    # sigmoid 內的方程式：(x-muj)/s
    phi = 1/(1+np.exp(-phi))    # sigmoid 
    return phi

data = pd.read_csv('E:/Machine Learning/HW2/1_data.csv')

# Bayesian Linear Regression 
beta = 1
M = 7
s = 0.1
N = [10,15,30,80]
x = np.asarray(data['x']).reshape(-1,1)
t = np.asarray(data['t']).reshape(-1,1)
j = np.arange(M).reshape(-1,1)
I = np.eye(len(j))
S0 = (10**-6)*I
m0 = 0

''' Q1-1 find mN & SN ''' 
phi = base_func(x,j,M,s)
SN = np.linalg.inv(S0 + beta*np.matmul(phi.T,phi))
mN = beta * np.matmul(np.matmul(SN,phi.T),t)

''' Q1-2 & 3''' 
# con：continuous x in range 0-4 (要連續數值才可以畫出線條)
con = np.arange(0, 4.01, 0.01)
con = base_func(np.asarray(con).reshape(-1,1), j, M, s)
for i in range(4):
    sample = data[:N[i]]
    # transfer sample_x into phi matrix
    sample_x = base_func(np.asarray(sample['x']).reshape(-1,1), j, M, s)
    SN = np.linalg.inv(S0 + beta*np.matmul(sample_x.T,sample_x))
    mN = beta * np.matmul(np.matmul(SN,sample_x.T),sample['t'])
    # sample 5 w
    w = np.random.multivariate_normal(mN.reshape((-1,)),SN,5)
    # make the distribution under sample_x & 5 w 
    con_y = con.dot(w.T)
    sample_y = sample_x.dot(w.T)
    ''' Q1-2 plot the distribution with 5 different w'''
    # plot the distribution of what we sampled and set x axis in range(0 - 4,0.01)
    plt.plot(sample['x'],sample['t'],'.')
    plt.plot(np.arange(0, 4.01, 0.01),con_y)   
    plt.show()
    
    ''' Q1-3 plot the distribution with (x,t) & mean curve'''
    mean = mN.reshape(-1,1).T.dot(con.T).reshape((-1,))
    std = 1/beta + np.diag(con.dot(SN).dot(con.T))
    plt.plot(sample['x'],sample['t'],'.')
    plt.plot(np.arange(0, 4.01, 0.01),mean,'b')
    plt.fill_between(np.arange(0, 4.01, 0.01), mean+std ,mean-std,color='lightblue')    
    plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' Q2-1 '''
train = pd.read_csv('E:/Machine Learning/HW2/train.csv',header=None)    # The data does'nt have header
test = pd.read_csv('E:/Machine Learning/HW2/test.csv',header=None)
x = train.iloc[:,3:].values
t = train.iloc[:,0:3].values

w0 = np.zeros((3,7))
ak = x.dot(w0.T)
yk = np.exp(ak) / np.sum(np.exp(ak), axis=1).reshape((-1,1))

def update_w(wo,x,y,t):
    wn = np.zeros((3,x.shape[1]))
    H1, H2, H3 = H(x,y)
    dE1, dE2, dE3 = delta_E(x,y,t)

    wn[0] = wo[0] - H1.dot(dE1).T
    wn[1] = wo[1] - H2.dot(dE2).T
    wn[2] = wo[2] - H3.dot(dE3).T
    y[:,0] = x.dot(wn[0].T)
    y[:,1] = x.dot(wn[1].T)
    y[:,2] = x.dot(wn[2].T)
    y = softmax(y)
    error = cross_entropy(y,t)
    return wn,y,error

def H(x,y):
    R = y*(1-y)
    R1, R2, R3 = np.zeros((180,180)),np.zeros((180,180)),np.zeros((180,180))
    np.fill_diagonal(R1, R[:,0])
    np.fill_diagonal(R2, R[:,1])
    np.fill_diagonal(R3, R[:,2])

    Hfunc1 = np.linalg.pinv(x.T.dot(R1).dot(x))
    Hfunc2 = np.linalg.pinv(x.T.dot(R2).dot(x))
    Hfunc3 = np.linalg.pinv(x.T.dot(R3).dot(x))
    return Hfunc1, Hfunc2, Hfunc3

def softmax(y):
    y = np.exp(y) / np.sum(np.exp(y), axis=1).reshape((-1,1))
    return y

def delta_E(x,y,t):
    dE1 = x.T.dot(y[:,0]-t[:,0])
    dE2 = x.T.dot(y[:,1]-t[:,1])
    dE3 = x.T.dot(y[:,2]-t[:,2])
    return dE1, dE2, dE3

# calculate the cross entropy
def cross_entropy(y,t):
    error = -np.sum(t * np.log(y)) 
    return error

# calculate the accuracy
def Accuracy(y,t):
    #pre = np.round(y)
    pre = np.zeros_like(y)
    pre[np.arange(len(y)), y.argmax(1)] = 1    
    acc = 1-(np.count_nonzero(pre-t)/pre.shape[0])
    return acc

# Initial y & w
y = yk.copy()
wn, y,error = update_w( w0, x, yk, t)
acc = []
err = []
while( error > 0.005):
    wn, y, error = update_w( wn, x, y, t)
    acc.append(Accuracy(y,t))
    err.append(error)
# plot accuracy matrix
fig,ax1 = plt.subplots()
ax2 = ax1.twinx() 
ax1.plot(acc,'b')
ax2.plot(err,color='orange')

ax1.set_xlabel('epochs')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.title('Accuracy rate & loss')
fig.tight_layout()
plt.show()

#%%
''' Q2-2 '''
test_x = test.values
predict = np.round(softmax(test_x.dot(wn.T)))

#%%
''' Q2-3 '''
for i in range(x.shape[1]):
    plt.hist(x[:60,i],color = 'salmon',alpha=0.7)
    plt.hist(x[60:120,i],color = 'yellowgreen',alpha=0.7)
    plt.hist(x[120:,i],color = 'lightblue',alpha=0.7)
    plt.show()
    
#%%
''' Q2-5 '''
plt.plot(x[:60, 0], x[:60, 1], '.')
plt.plot(x[60:120, 0], x[60:120, 1], '.')
plt.plot(x[120:, 0], x[120:, 1], '.')
plt.show()

#%%
''' Q2-6 '''
# Initial y & w
x2 = x[:,:2]
wn2 = np.zeros((3,2))
y2 = softmax(x2.dot(wn2.T))
error2 = 1
acc2 = []
err2 = []
while(error2>0.005):
    wn2, y2, error2 = update_w( wn2, x2, y2, t)
    acc2.append(Accuracy(y2,t))
    err2.append(error2)
    if (len(err2) > 1 and err2[-2]-error2<0.001):
        break
# plot accuracy matrix
fig,ax1 = plt.subplots()
ax2 = ax1.twinx() 
ax1.plot(acc2,'b')
ax2.plot(err2,color='orange')

ax1.set_xlabel('epochs')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.title('Accuracy rate & loss')
fig.tight_layout()
plt.show()

test_x = test.values
predict2 = np.round(softmax(test_x.dot(wn.T)))

#%%
''' Q2-7 '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('E:/Machine Learning/HW2/train.csv',header=None)
test = pd.read_csv('E:/Machine Learning/HW2/test.csv',header=None)
x = train.iloc[:,3:].values
t = train.iloc[:,0:3].values

x1 = x[:60]
x2 = x[60:120]
x3 = x[120:]

m = np.mean(x,axis=0)
m1 = np.mean(x1,axis=0)
m2 = np.mean(x2,axis=0)
m3 = np.mean(x3,axis=0)
SW = (x - m1).T.dot((x - m1)) + (x - m2).T.dot((x - m2)) + (x - m3).T.dot((x - m3))

M1 = (m1 - m).reshape((-1,1))
M2 = (m2 - m).reshape((-1,1))
M3 = (m3 - m).reshape((-1,1))

SB = 60*(M1.dot(M1.T)) + 60*(M2.dot(M2.T)) + 60*(M3.dot(M3.T)) 
ST = SW +SB

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))
eig_pairs = []
for i in range(len(eig_vals)):
    eig_pairs.append((np.abs(eig_vals[i]), eig_vecs[:,i]))
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W = np.hstack((eig_pairs[0][1].reshape(7,1), eig_pairs[1][1].reshape(7,1)))
y = x.dot(W)

plt.plot(y[:60,0],y[:60,1],'.',color = 'salmon')
plt.plot(y[60:120,0],y[60:120,1],'.',color = 'yellowgreen')
plt.plot(y[120:,0],y[120:,1],'.',color = 'lightblue')

plt.grid()
plt.tight_layout
plt.show()
#%%
''' Q3-1 '''
def Accuracy_Rate(y,t):
    A = 1 - (np.count_nonzero(t-y) / len(y))
    return A

def normalize(x):
    std = np.std(x,axis=0).reshape((-1,1))
    mean = np.mean(x,axis=0).reshape((-1,1))
    normalized_x = (x - mean.T) / std.T
    return normalized_x

def KNN(test,train,target,K):
    test_num = test.shape[0]
    train_num = train.shape[0]
    _test = np.repeat(test, train_num, axis=0)
    _train = np.tile(train,( test_num,1))
    dist = np.sqrt(np.sum(np.square(_test - _train),axis=1)).reshape((60,150)).T
    
    arg_sort = np.argsort(dist, axis=0)[:K]
    ans = target[arg_sort].astype(np.int32)
    y = np.zeros((test_num, ))
    for i in range(test_num):
        y[i] = np.bincount(ans[:, i]).argmax()
    return y

seeds = pd.read_csv('E:/Machine Learning/HW2/seeds.csv').values
# normalize the data
x = seeds[:,:7]
t = seeds[:,-1].reshape((-1,1))
x_nor = normalize(x)

# split data into trainning data and testing data
data = np.concatenate((x_nor,t),axis=1)
train = np.concatenate((data[:50],data[70:120],data[140:190]),axis=0)
train_x = train[:,:7]
train_y = train[:,-1]

test = np.concatenate((data[50:70],data[120:140],data[190:210]),axis=0)
test_x = test[:,:7] 
test_y = test[:,-1]

Ans, Acc = [], []
K = np.arange(1,11)
for k in K:
    Ans.append(KNN(test_x,train_x,train_y,k))
    Acc.append(Accuracy_Rate(Ans[k-1],test_y))
import matplotlib.pyplot as plt
plt.plot(Acc)
plt.show()
#%%
''' Q3-2 '''
def KNN_dist(test,train,target,distence):
    test_num = test.shape[0]
    train_num = train.shape[0]
    _test = np.repeat(test, train_num, axis=0)
    _train = np.tile(train,( test_num,1))
    dist = np.sqrt(np.sum(np.square(_test - _train),axis=1)).reshape((60,150)).T

    y = np.zeros((test_num, ))
    
    for i in range(test_num):
        pred = target[dist[:, i] < distence].astype(np.int32)
        y[i] = np.bincount(pred).argmax()
    return y

train_x = train[:,:7]
train_y = train[:,-1]
test_x = test[:,:7] 
test_y = test[:,-1]

Ans, Acc = [], []
V = np.arange(2,11)

for v in V:
    Ans.append(KNN_dist(test_x,train_x,train_y,v))
    Acc.append(Accuracy_Rate(Ans[v-2],test_y))
import matplotlib.pyplot as plt
plt.plot(Acc)
plt.show()