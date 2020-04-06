# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:26:41 2018

@author: Susan
"""

import pandas as pd 
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt

data = pd.read_csv('E:/Machine Learning/HW1/housing.csv')
s = np.arange(len(data))
np.random.shuffle(s)
data = data.iloc[s]

data = data.dropna()
train_x = data.iloc[:int(len(data) * 0.9),:3]
train_y =  data.iloc[:int(len(data) * 0.9),-1]

test_x = data.iloc[int(len(data) * 0.9):,:3]
test_y =  data.iloc[int(len(data) * 0.9):,-1]

class Regressor:
    def fit(self,x,y,λ):
        if λ==0:
            self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
            return self.w
        else:
            i = np.eye(x.shape[1])
            self.wrls = np.matmul(np.matmul(np.linalg.inv(λ*i+np.matmul(x.T,x)),x.T),y)
            return self.wrls

    # make the Phi matrix (hand draft p.42)
    def intersection(self,x,M):
        m = np.ones((len(x),1))
        for i in range(M):
            for combination in list(itertools.combinations_with_replacement(range(len(x.columns)), i+1)):
                tx = x.values
                mm = np.prod(tx[:,combination],axis=1).reshape(-1,1)
                m = np.concatenate((m,mm),axis=1)
        return m

    def predict(self,x,λ):
        if λ==0:
            return np.matmul(self.w.T, x.T)
        else:
            return np.matmul(self.wrls.T, x.T) 
    
    def RMSE(self,x,y,λ):
        if λ==0:
            Erms = 0.5*sum((self.predict(x,λ)-y)**2)
            Erms = math.sqrt(2*Erms/len(x))
            return Erms
        else:
            Erms_wrls = 0.5*sum((self.predict(x,λ)-y)**2)+0.5*λ*np.matmul(self.wrls.T,self.wrls)
            Erms_wrls = math.sqrt(2*Erms_wrls/len(x))
            return Erms_wrls

# Q1 make three multiple regession
# Q3 Compare the result with different λ
λ = 0       #Q1
#λ = 0.1     #Q3-1
#λ = 0.001   #Q3-2
reg3 = Regressor()
train_x3 = reg3.intersection(train_x, 3)
test_x3 = reg3.intersection(test_x, 3)
reg3.fit(train_x3, train_y,λ)
prediction = reg3.predict(test_x3,λ)
error3 = reg3.RMSE(test_x3,test_y,λ)
error3_train = reg3.RMSE(train_x3,train_y,λ)
del train_x3,test_x3

reg2 = Regressor()
train_x2 = reg2.intersection(train_x, 2)
test_x2 = reg2.intersection(test_x, 2)
reg2.fit(train_x2, train_y,λ)
prediction = reg2.predict(test_x2,λ)
error2 = reg2.RMSE(test_x2,test_y,λ)
error2_train = reg2.RMSE(train_x2,train_y,λ)
del train_x2,test_x2

reg = Regressor()
train_x1 = reg.intersection(train_x, 1)
test_x1 = reg.intersection(test_x, 1)
reg.fit(train_x1, train_y,λ)
prediction = reg.predict(test_x1,λ)
error = reg.RMSE(test_x1,test_y,λ)
error_train = reg.RMSE(train_x1,train_y,λ)
del train_x1,test_x1

# Q1 plot the outcome of error under different M and dataset
train = [error_train,error2_train,error3_train]
test = [error,error2,error3]
M = [1,2,3]

plt.xticks(M)
plt.plot(M,train,marker='o',label='train')
plt.plot(M,test,marker='o',label='test')

plt.xlabel('M')
plt.legend()
plt.title("RMSE")
plt.show()

# Q3 the outcome under different λ
if λ!=0:
    print('error3_%.3f:%f'%(λ,error3))
    print('error3_train_%.3f:%f'%(λ,error3_train))
    print('error2_%.3f:%f'%(λ,error2))    
    print('error2_train_%.3f:%f'%(λ,error2_train))
    print('error_%.3f:%f'%(λ,error))
    print('error_train_%.3f:%f'%(λ,error_train))

# Q2 select the most contributive attribute
reg = Regressor()
train_x_mi = reg.intersection(train_x[['total_rooms','population']], 3)
test_x_mi = reg.intersection(test_x[['total_rooms','population']], 3)
reg.fit(train_x_mi, train_y,0)
prediction = reg.predict(test_x_mi,0)
error_without_mi = reg.RMSE(train_x_mi,train_y,0)
del train_x_mi,test_x_mi

reg = Regressor()
train_x_to = reg.intersection(train_x[['median_income','population']], 3)
test_x_to = reg.intersection(test_x[['median_income','population']], 3)
reg.fit(train_x_to, train_y,0)
prediction = reg.predict(test_x_to,0)
error_without_to = reg.RMSE(train_x_to,train_y,0)
del train_x_to,test_x_to

reg = Regressor()
train_x_po = reg.intersection(train_x[['median_income','total_rooms']], 3)
test_x_po = reg.intersection(test_x[['median_income','total_rooms']], 3)
reg.fit(train_x_po, train_y,0)
prediction = reg.predict(test_x_po,0)
error_without_po = reg.RMSE(train_x_po,train_y,0)
del train_x_po,test_x_po

print('error without median_income:%f'%error_without_mi+'\n'+'error without total_rooms:%f'%error_without_to+'\n'+'error without population:%f'%error_without_po)