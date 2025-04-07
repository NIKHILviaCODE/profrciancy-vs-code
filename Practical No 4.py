#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Aim :- To perform and analysis of Z Test parametric Test


# In[1]:


#Name :-  Nikhil Anil Kakar
#Roll No.:- 52
#Sec :- A
#Subject :- BDA(ET-2)


# In[2]:


# Python program to implement One Sample Z-Test   
  
# Importing the required libraries  
import pandas as pd  
from scipy import stats  
from statsmodels.stats import weightstats as stests  
               
    
# Creating a dataset  
data = [89, 93, 95, 93, 97, 98, 96, 99, 93, 97,  
        110, 104, 119, 105, 104, 110, 110, 112, 115, 114,26,65,76,87,98,23,45,67,89,90,87,76,65,54,43,32,21,67,78,79]  
  
# Performing the z-test  
z_test ,p_val = stests.ztest(data)  
print(p_val)  
  
# taking the threshold value as 0.05 or 5%  
if p_val < 0.05:  
    print("We can reject the null hypothesis")  
else:  
    print("We can accept the null hypothesis")  


# In[ ]:




