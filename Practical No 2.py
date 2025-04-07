#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Aim :- To perform and analysis for Normal Distribution in given dataset


# In[ ]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[1]:


#importing the basic library
import pandas as pd 


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop')


# In[5]:


data=pd.read_csv("diabetes.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.shape


# In[11]:


data.size


# In[12]:


data.ndim


# In[13]:


data.columns


# In[14]:


data.isna()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[18]:


sns.distplot(data,bins=20)
plt.show()


# In[26]:


sns.distplot(data['Glucose'],bins=20)
plt.show()


# In[27]:


sns.distplot(data['Age'],bins=20)
plt.show()


# In[32]:


sns.distplot(data['BloodPressure'],bins=20)
plt.show()


# In[33]:


sns.distplot(data['SkinThickness'],bins=20)
plt.show()


# In[34]:


import matplotlib.pyplot as plt


# In[37]:


plt.hist(data['Age'], bins=30, color='blue', edgecolor='black', alpha=0.7)


# In[ ]:




