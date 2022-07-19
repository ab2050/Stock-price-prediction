#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv(r'E:\AB\MACHINE LEARNING THINGS\Linear Regression\NY stock exchange\prices-split-adjusted.csv')


# In[2]:


data.head()


# In[3]:


df=pd.DataFrame(data)
df.describe()


# In[31]:


dfr=df.drop('date',axis=1)


# In[32]:


#dfr = dfr.drop('close',axis=1)
#dfr = dfr.drop('symbol',axis=1)
dfr


# In[33]:


y = df['close']
y


# In[34]:


#dfr = dfr.head(1000)
dfr


# In[21]:


df=df.drop('close',axis=1)


# In[35]:


X = dfr
X


# In[25]:


y


# In[36]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()


# In[37]:


w = np.array([1,1,1,1])
b=2
w.shape


# In[38]:


X = X.to_numpy()
X


# In[50]:


X[1,2]


# In[39]:


n=X.shape[0]
n


# In[44]:


xve = X[0,:]
xve


# In[40]:


def cost_func(X,y,w,b):
    
    
    cf=0
    fwb=0
    
    m = X.shape[0]
    for i in range(m):
        fwb= np.dot(X[i],w)+b
        cf += (fwb-y[i])**2
    cf = cf/(2*m)
    return cf


# In[41]:


print(f'Cost function is : {cost_func(X,y,w,b)} ')


# In[28]:


import math


# In[109]:


#Since cost function is very high, gradient descent will be done next
#w.shape
X[2].shape


# In[42]:


def deriv(X,y,w,b):
    
    m = X.shape[0]
    n = X.shape[1]
    dw = np.zeros((n,))
    db = 0.
    cf=0
    
    for i in range(m):
        fwb = np.dot(X[i],w)+b 
        cf = fwb-y[i]
        for j in range(n):
            dw[j]+= cf*X[i][j]
        db+=cf
    dw=dw/m
    db=db/m
    
    return dw,db


# In[53]:


def grad_desc(X,y,w,b,iters):
    
    gd=0
    cf=0
    fwb=0
    ap = 0.42
    dw=0
    db=0
    hist = []
    
    for i in range(iters):
        dw,db = deriv(X,y,w,b)
        w = w-(ap*dw)
        b = b-(ap*db)
        
        if(i<2000):
            hist.append(cost_func(X,y,w,b))
            
        if i% math.ceil((iters-1) / 10) == 0:
            print(f"Iteration {i:4d}: Cost {hist[-1]:8.2f}   ")
        
    return w,b,hist


# In[54]:


it = 1001
b=0
w_final, b_final, J_hist = grad_desc(X, y, w, b,it)                                         
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X.shape
for i in range(m):
    print(f"prediction: {np.dot(X[i], w_final) + b_final:0.2f}, target value: {y[i]}")


# In[144]:


#NEEDS TO BE REDONE NEW VALUES NOT UPDATED !!
t=np.dot(X[2], w_final)


# In[145]:


"""cost function much higher for 800,000 dataset as compared to 10,000 one for same learning rate.
b,w found by gradient descent: 32.75,[268.91654216 272.18242873 268.68700124  -9.51587609] 
final cost on iteration 1000 : 668
prediciton pretty far off
Learning rate : 0.105
-----------------------
Learning rate = 0.28
Final cost on iteration 1000 : 62
-----------------------
Learning rate = 0.32
Final cost on iteration 1000 : 36.18
-----------------------
Learning rate = 0.4
Final cost on iteration 1000 : 12.45


# In[55]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[56]:


yh = []
yav = []
for i in range(250):
    yh.append(np.dot(X[i],w_final)+b)
    t=y[i]
    yav.append(t)
yh


# In[57]:


ya = np.array(yh)
yar = np.array(yav)


# In[58]:


plt.plot(ya, label = "Predicted")#, linestyle="-")
plt.plot(yar, label = "Actual")#, linestyle="--")
#plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
#plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")
plt.legend()
plt.show()


# In[ ]:




