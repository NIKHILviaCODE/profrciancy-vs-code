#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of Naive Bayes Algorithm

# # Importing the Libraries

# In[ ]:


#Aim :- To perform and analysis of Naive Bayes, confusion matrix, K fold Cross Validation


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


# In[5]:


data=pd.read_csv("heart.csv")


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


# # Data preprocessing _ data cleaning _ missing value treatment

# In[13]:


# check Missing Value by record 

data.isna()


# In[14]:


data.isna().any()


# In[15]:


data.isna().sum()


# # Removing duplicates 

# In[16]:


data_dup =data.duplicated().any()


# In[17]:


data_dup


# In[18]:


data=data.drop_duplicates()


# In[19]:


data_dup =data.duplicated().any()


# In[20]:


data_dup


# # Splitting of DataSet into train and Test

# In[21]:


x=data.drop("target", axis=1)
y=data["target"]


# In[22]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[23]:


x_train


# In[24]:


x_test


# In[25]:


y_train


# In[26]:


y_test


# # Naive Bayes classifier

# In[27]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 


# In[28]:


nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)


# In[29]:


y_pred = nb_classifier.predict(x_test)


# In[30]:


accuracy_score (y_test,y_pred)


# # confusion matrix

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Reds', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[33]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score


# In[44]:


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

# Recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall:.4f}')

# Error Rate
error_rate = 1 - accuracy
print(f'Error Rate: {error_rate:.4f}')

# Classification report
print("Classification Report:")
print(classification_report(y_test,y_pred))


# In[ ]:





# # K fold Cross Validation

# In[34]:


from sklearn.model_selection import KFold, cross_val_score


# In[35]:


# Define K-Fold Cross Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform Cross Validation
scores = cross_val_score(nb_classifier, x, y, cv=kf, scoring='accuracy')

# Print results
print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {scores.mean():.4f}')


# In[ ]:





# In[ ]:





# In[ ]:




