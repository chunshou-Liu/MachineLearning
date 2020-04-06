# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:58:18 2018

@author: Susan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('E:/Machine Learning/HW3/problem1/gp.csv',header=None)
x = data[0].values
t = data[1].values

# Gaussian Process
class GP:
    def __init__(self, theta,beta):
        self.theta = theta
        self.beta = beta
        
    def kernel(self,xn,xm):
        K_A = self.theta[0]*(np.exp((-self.theta[1]/2)*self.euclid(xn,xm))) + self.theta[2] + self.theta[3]*(xm.reshape((-1,1)).dot(xn.reshape((-1,1)).T))
        return K_A
    
    def euclid(self,xn,xm):
        test_copy_times = xm.shape[0]
        train_copy_times = xn.shape[0]
        _test = np.repeat(xn, test_copy_times, axis=0)
        _train = np.tile(xm,train_copy_times)
        dist = np.sum(np.square(_test - _train).reshape(-1,1),axis=1).reshape((test_copy_times,test_copy_times)).T
        return dist
    
    def predict(self, train_x, train_y, test_x):
        deltamn = np.eye((train_x.size))
        self.CN = self.kernel(train_x, train_x) + beta*(deltamn) # CN
        self.K = self.kernel(train_x, test_x) # K
        self.c = self.kernel(test_x, test_x) + beta*(deltamn) # c
    
        self.mu = self.K.dot(np.linalg.inv(self.CN)).dot(train_y)
        var = self.c - self.K.dot(np.linalg.inv(self.CN)).dot(self.K.T)
        return self.mu,np.diag(var)
    
    def RMSE(self, test_y):
        self.rmse = np.sqrt(np.mean((self.mu - test_y)**2))
        return self.rmse


theta = [[1,4,0,0],[0,0,0,1],[1,4,0,5],[1,64,10,0]]
train_x, train_y = x[:60],t[:60]
test_x , test_y = x[60:],t[60:]
beta = 1

'''Q1：Gaussian Process'''
for i in range(4):
    gp = GP(theta[i], beta)
    mu,var = gp.predict(train_x, train_y, np.linspace(0,2,60))
    plt.plot(x,t,'.')
    plt.plot(np.linspace(0,2,60),mu.reshape(-1,1),'r')
    plt.fill_between(np.linspace(0,2,60), mu+var ,mu-var,color='pink')
    plt.title('theta = '+ str(theta[i]))
    plt.show()
    mu,var = gp.predict(train_x, train_y, train_x)
    train_RMSE = gp.RMSE(train_y)
    mu,var = gp.predict(train_x, train_y, test_x)
    test_RMSE = gp.RMSE(test_y)
    print('【RMSE】\ntrain：%f \ntest：%f' %(train_RMSE, test_RMSE))
#%%
'''Q2：Support Vector Machine'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC

x = pd.read_csv('E:/Machine Learning/HW3/problem2/x_train.csv',header=None).values
t = pd.read_csv('E:/Machine Learning/HW3/problem2/t_train.csv',header=None).values

# Support Vector Machine
class SVM:
    def __init__(self,input_type):
        self.type_ = input_type
    
    def linear_kernel(self,x):
        self.lk = x.dot(x.T)
        return self.lk
    
    def polynomial_kernel(self,x):
        phi = self.phi_process(x)
        self.pk = phi.dot(phi.T)
        return self.pk
        
    def phi_process(self,x):
        x1 = x[:,0]
        x2 = x[:,1]
        self.phi = np.zeros((x1.shape[0],3))
        self.phi[:,0] = x1**2
        self.phi[:,1] = (2**0.5)*x1*x2
        self.phi[:,2] = x2**2
        return self.phi
    
    def fit(self,x,t):
        self.c1 = np.min(t)
        self.c2 = np.max(t)
        t[t == self.c1] = 1
        t[t == self.c2] = -1
        t = t.reshape(-1,)
        
        if self.type_ == 0: # linear
            svc = SVC(kernel='linear').fit(x,t)
            self.alpha = np.abs(svc.dual_coef_.reshape(-1,))
            self.spv = svc.support_
            self.w = np.sum(self.alpha * t[self.spv] * x[self.spv].T,axis = 1)
            self.b = np.mean(t[self.spv] - self.alpha * t[self.spv] * self.linear_kernel(x[self.spv]))
            #print(SVC(kernel='linear').fit(x, t).coef_) # 對答案
        else:  # polynomial_kernel
            svc2 = SVC(kernel='poly',degree = 2).fit(x,t)
            self.alpha2 = np.abs(svc2.dual_coef_.reshape(-1,))
            self.spv2 = svc2.support_
            self.w = np.sum(self.alpha2 * t[self.spv2] * self.phi_process(x[self.spv2]).T, axis = 1)
            self.b = np.sum((t[self.spv2] - self.alpha2 * t[self.spv2] * self.polynomial_kernel(x[self.spv2])))/len(self.spv2)
            
        return self.w
    
    def predict(self,x):
        if self.type_ == 0 :
            self.y = self.w.dot(x.T) + self.b
        else:
            self.y = self.w.dot(self.phi_process(x).T) + self.b
        self.predict_ = np.zeros_like(self.y)
        self.predict_[self.y > 0 ] = self.c1
        self.predict_[self.y <= 0] = self.c2
        return self.predict_.reshape(-1,1)
    
    def vote(self,pre_14,pre_49,pre_19):
        ans_array = np.concatenate((pre_14,pre_49,pre_19),axis = 1).astype(int)
        self.ans = np.zeros_like(pre_14)
        for i in range(ans_array.shape[0]):
            self.ans[i] = np.argmax(np.bincount(ans_array[i]))
        return self.ans
    
# split data to fit one-versus-one
x14,t14 = x[:100].copy(), t[:100].copy()
x49,t49 = x[50:].copy(), t[50:].copy()
x19 = np.concatenate((x[:50].copy().reshape(-1,2),x[100:].copy().reshape(-1,2)),axis = 0)
t19 = np.concatenate((t[:50].copy().reshape(-1,1),t[100:].copy().reshape(-1,1)),axis = 0)

# drawing data
y = np.vstack((np.linspace(-1,1,200),np.linspace(-1,1,200))).T
color = ['red', 'blue' ,'limegreen']
xx, yy = np.meshgrid(y[:,0],y[:,1])
xx, yy = np.ravel(xx),np.ravel(yy)
xy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)),axis=1)

kernel_type = 1
svm = SVM(kernel_type)

svm14  = svm.fit(x14.copy(),t14.copy())
if kernel_type == 0:
    spv0_14 = svm.spv
else:
    spv1_14 = svm.spv2
predict_14 = svm.predict(x)
Z1 = svm.predict(xy)

svm49  = svm.fit(x49.copy(),t49.copy())
if kernel_type == 0:
    spv0_49 = svm.spv
else:
    spv1_49 = svm.spv2
predict_49 = svm.predict(x)
Z2 = svm.predict(xy)

svm19  = svm.fit(x19.copy(),t19.copy())
if kernel_type == 0:
    spv0_19 = svm.spv
else:
    spv1_19 = svm.spv2
predict_19 = svm.predict(x)
Z3 = svm.predict(xy) 

A = svm.vote(predict_14,predict_49,predict_19)
Z = svm.vote(Z1,Z2,Z3)
Z = Z.reshape(200,200)
xx, yy = xx.reshape(200,200),yy.reshape(200,200)

# Draw SVM
plt.figure()
plt.contourf(xx, yy, Z,2, alpha=0.3, antialiased=True, colors = color)

plt.scatter(x[:50,0], x[:50,1], color ='red', label='Class 1')
plt.scatter(x[50:100,0], x[50:100,1], color ='blue', label='Class 4')
plt.scatter(x[100:,0], x[100:,1], color ='limegreen', label='Class 9')

if kernel_type == 0 :
    plt.scatter(x14[spv0_14,0], x14[spv0_14,1],color ='none', edgecolors='k')
    plt.scatter(x49[spv0_49,0], x49[spv0_49,1],color ='none', edgecolors='k')
    plt.scatter(x19[spv0_19,0], x19[spv0_19,1],color ='none', edgecolors='k')
else:
    plt.scatter(x14[spv1_14,0], x14[spv1_14,1],color ='none', edgecolors='k')
    plt.scatter(x49[spv1_49,0], x49[spv1_49,1],color ='none', edgecolors='k')
    plt.scatter(x19[spv1_19,0], x19[spv1_19,1],color ='none', edgecolors='k')

plt.legend()
plt.title('Linear' if kernel_type == 0 else 'Polynomial')
plt.show()
#%%
'''Q3'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

I = Image.open("E:/Machine Learning/HW3/problem3/hw3.jpg")
data = np.asarray(I)

#plt.imshow(I)

class Kmeans:
    def __init__(self,k):
        self.k = k
    
    def random_k(self,x):        
        return x[np.random.randint(0,x.shape[0],size=self.k)]
    
    def euclid(self,xn,xm):
        print(xm.shape,xn.shape)
        test = np.repeat(xn, xm.shape[0],axis=0)
        self.B = test
        train = np.tile(xm, (xn.shape[0],1))
        self.A = train
        dist = np.sqrt(np.sum((train - test)**2, axis=1)).reshape(xn.shape[0],xm.shape[0])
        return dist
    
    def fit(self,x):
        centers = self.random_k(x)
        for _ in range(10):
            dist = self.euclid(centers,x)
            print(dist.shape)
            group = np.argmin(dist,axis=0)
            for i in range(self.k):
                centers[i] = np.mean(x[group == i], axis=0)
        return centers,group

K = 50
km = Kmeans(K)
center, group = km.fit(data.reshape(-1,3))
ans = np.zeros_like(data.reshape(-1,3)) 
for i in range(K):
    ans[ group == i ] = center[i]
plt.imshow(ans.reshape(data.shape))