#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Aim:- To Find Unique and Duplicates Value Count in given dataset


# In[ ]:


#Name :- Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[2]:


#importing the basic library
import pandas as pd 


# In[3]:


import os


# In[4]:


os.getcwd()


# In[39]:


os.chdir('C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop')


# In[24]:


data=pd.read_csv("diabetes.csv")


# In[25]:


data.head()


# In[26]:


data.tail()


# In[27]:


data.info()


# In[28]:


data.describe()


# In[29]:


data.shape


# In[30]:


data.size


# In[31]:


data.ndim


# In[32]:


data.columns


# In[33]:


data.isna()


# In[34]:


data.isna().any()


# In[35]:


data.isna().sum()


# In[36]:


data['Age'].unique()


# In[37]:


data['Age'].duplicated()


# In[38]:


data['Age'].duplicated().sum()

