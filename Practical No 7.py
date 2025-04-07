#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Aim : To perform and analysis of logistic Regression Algorithm


# In[2]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import os


# In[5]:


os.getcwd()


# In[7]:


os.chdir("C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop")


# In[8]:


df = pd.read_csv("framingham.csv")


# In[9]:


df.head()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


df.isna().sum()


# In[13]:


df


# # Missing Value Treatment

#  Since,'glucose' and 'education' columns had a significant amount of all nul values,so we
#  replaced them with the mean of values for their respective columns

# In[14]:


df['glucose'].fillna(value = df['glucose'].mean(),inplace=True)


# In[15]:


df['education'].fillna(value = df['education'].mean(),inplace=True)


# In[16]:


df['heartRate'].fillna(value = df['heartRate'].mean(),inplace=True)


# In[17]:


df['BMI'].fillna(value = df['BMI'].mean(),inplace=True)


# In[18]:


df['cigsPerDay'].fillna(value = df['cigsPerDay'].mean(),inplace=True)


# In[19]:


df['totChol'].fillna(value = df['totChol'].mean(),inplace=True)


# In[20]:


df['BPMeds'].fillna(value = df['BPMeds'].mean(),inplace=True)


# In[21]:


df.isna().sum()


# In[22]:


#Splitting the dependent and independent variables.
x = df.drop("TenYearCHD",axis=1)
y = df['TenYearCHD']


# In[23]:


x #checking the features


# # Train Test Split

# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[27]:


y_train


# # Logistic Regression Algorithm

# In[28]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train,y_train)
model.score(x_train, y_train)


# In[ ]:




