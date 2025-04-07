#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# In[ ]:


#Aim:- To Perform and analysis of Linear Regression Algorithm


# In[ ]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop')


# In[5]:


df = pd.read_csv("heart.csv")


# In[6]:


df


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.size


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.isnull()


# In[14]:


df.isnull().any()


# In[15]:


df.isnull().sum()


# In[16]:


a = "ashish"


# In[17]:


print(a)


# In[18]:


a[0]


# In[19]:


a[-1]


# In[20]:


a[1:3]


# In[21]:


a[1:4]


# In[22]:


#Assiging values in X & Y

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#X = df['YearsExperience']
#y = df['Salary']


# In[23]:


import matplotlib.pyplot as plt


# In[24]:


import seaborn as sns
import numpy as np


# In[25]:


print(X)


# In[26]:


print(y)


# In[27]:


#Splitting testdata into x_train,x_test,y_train,y_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42) 


# In[28]:


print(X_train)


# In[29]:


print(X_test)


# In[30]:


print(y_train)


# In[31]:


print(y_test)


# In[32]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[33]:


#Assigning Coefficient (slope) to m
m = lr.coef_


# In[34]:


print("Coefficient :", m)


# In[35]:


#Assigning Y-intercept to a
c = lr.intercept_


# In[36]:


print("Intercept : ", c)


# In[37]:


lr.score(X_test,y_test) * 100


# In[ ]:





# In[ ]:




