#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv(r'E:\AB\MACHINE LEARNING THINGS\Linear Regression\NY stock exchange\prices-split-adjusted.csv')


# In[2]:


df=pd.DataFrame(data)
df


# In[3]:


df=df.drop('symbol',axis=1)
dfr = df


# In[7]:


dfr=dfr.drop('close',axis=1)
dfr


# In[8]:


y = df['close']
y


# In[89]:


dfr=dfr.drop('date',axis=1)


# In[9]:


dfr


# In[16]:


y


# In[20]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(dfr), columns=dfr.columns)
X.head()


# In[21]:


w = np.array([1,1])
b=2
X


# In[22]:


X = X.to_numpy()
X


# In[23]:


def cost_func(X,y,w,b):
    
    cf=0
    fwb=0
    
    m = X.shape[0]
    for i in range(m):
        fwb= np.dot(X[i],w)+b
        cf += (fwb-y[i])**2
    cf = cf/(2*m)
    return cf


# In[24]:


import math
print(f'Cost function is : {cost_func(X,y,w,b)} ')


# In[25]:


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


# In[26]:


def grad_desc(X,y,w,b,iters):
    
    gd=0
    cf=0
    fwb=0
    ap = 1.98
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


# In[27]:


it = 1001
b=0
w_final, b_final, J_hist = grad_desc(X, y, w, b,it)                                         
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X.shape
for i in range(m):
    print(f"prediction: {np.dot(X[i], w_final) + b_final:0.2f}, target value: {y[i]}")


# In[ ]:


#model overshooting minima on alpha = 2 but 1.9 gives very high cost function
""" iterations : 1001
1.98 = 1.45


# In[28]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[72]:


y[5]


# In[29]:


yh = []
yav = []
for i in range(250):
    yh.append(np.dot(X[i],w_final)+b)
    t=y[i]
    yav.append(t)
yh


# In[30]:


ya = np.array(yh)
yar = np.array(yav)


# In[31]:


plt.plot(ya, label = "Predicted")#, linestyle="-")
plt.plot(yar, label = "Actual")#, linestyle="--")
#plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
#plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")
plt.legend()
plt.show()


# In[ ]:


"""
Note : model improves with increased alpha
DOUBT : MODEL IS GIVING ALMOST SIMILAR VLAUES FOR 3 PARAMETERS COULD THAT INDICATE A POSSIBLE ERROR ?
DATASET SIZE : FIRST 10,000 USING OPEN, LOW HIGH AND VOLUME
COST FUNCTIONS :
0.01 = 
0.05 = 
0.08 = 
0.09 = 
0.095 = 
0.1 = 1.27
0.105 = 
0.108 = 

