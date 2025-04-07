#!/usr/bin/env python
# coding: utf-8

# # Finding parameters of Confusion Matrix

# # Importing the Libraries

# In[ ]:


#Aim :- To perform and Data analysis with Confusion Matrix


# In[ ]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[1]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Lenovo\\OneDrive\\Doc\\Desktop')


# In[6]:


data=pd.read_csv("heart.csv")


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.shape


# In[12]:


data.size


# In[13]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[14]:


# check Missing Value by record 

data.isna()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# # Splitting of DataSet into train and Test

# In[32]:


x=data.drop("target", axis=1)
y=data["target"]


# In[35]:


data.head()


# In[38]:


#splitting the data into training and testing data sets
#train is part of Machine learning
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[39]:


x_train


# In[40]:


x_test


# In[41]:


y_train


# In[42]:


y_test


# # Logistic Regression

# In[43]:


data.head()


# In[44]:


#LogisticRegression is part of Machine Learning
from sklearn.linear_model import LogisticRegression


# In[45]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[51]:


y_pred1=log.predict(x_test)


# In[52]:


from sklearn.metrics import accuracy_score 


# In[53]:


accuracy_score (y_test,y_pred1)


# In[59]:


accuracy_score (y_test,y_pred1) * 100


# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[68]:


cm = confusion_matrix(y_test, y_pred1)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




