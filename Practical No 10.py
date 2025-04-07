#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Aim :- To perform and Data analysis with Co-relation Matrix


# In[19]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[20]:


#importing the basic library
import pandas as pd 


# In[21]:


import os


# In[22]:


os.getcwd()


# In[24]:


os.chdir('C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop')


# In[25]:


data=pd.read_csv("diabetes.csv")


# In[26]:


data.head()


# In[27]:


data.tail()


# In[28]:


data.info()


# In[29]:


data.describe()


# In[30]:


data.shape


# In[31]:


data.size


# In[32]:


data.ndim


# In[33]:


data.columns


# In[34]:


data.isna()


# In[35]:


data.isna().any()


# In[36]:


data.isna().sum()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[38]:


#correlation
corr = data.corr()


# In[39]:


sns.heatmap(data.corr())


# In[45]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr())


# In[46]:


plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True)


# In[ ]:




