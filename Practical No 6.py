#!/usr/bin/env python
# coding: utf-8

# # One Way F-test(Anova) :-

# It tell whether two or more groups are similar or not based on their mean similarity and f-score.
# 
# Example : there are 3 different category of iris flowers and their petal width and need to check whether all 3 group are similar or not

# In[ ]:


#Aim :- To perform and analysis of ANOVA parametric Test


# In[1]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[2]:


import seaborn as sns
df1=sns.load_dataset('iris')


# In[3]:


df1.head()


# In[4]:


df1.tail()


# In[5]:


df_anova = df1[['petal_width','species']]


# In[6]:


import pandas as pd
grps = pd.unique(df_anova.species.values)


# In[7]:


grps


# In[8]:


d_data = {grp:df_anova['petal_width'][df_anova.species == grp] for grp in grps}


# In[9]:


d_data


# In[10]:


import scipy.stats as stats


# In[11]:


F, p = stats.f_oneway(d_data['setosa'], d_data['versicolor'], d_data['virginica'])


# In[12]:


print(p)


# In[13]:


if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

