#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# In[2]:


# Data manipulation and analysis
import pandas as pd

# Scientific computing
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Algorithms
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


file_path = 'C:/Users/user/Desktop/dataset.csv'  # Replace 'path_to_your_dataset' with the actual path
heart_data = pd.read_csv(file_path)


# In[4]:


heart_data


# In[30]:


heart_data.head(16)


# In[5]:


heart_data.drop('education', axis = 1, inplace = True)


# In[6]:


null_values = heart_data.isnull().sum()


# In[7]:


null_values


# In[34]:


heart_data.head()


# In[8]:


heart_data = heart_data.rename(columns={"BPMeds": "blood pressure medication", "totChol": "cholesterol level" })


# In[9]:


data_types = heart_data.dtypes


# In[10]:


data_types


# In[11]:


null_values = heart_data.isnull().sum()


# In[12]:


null_values


# ## Cleaning

# In[13]:


#remove non values


# In[43]:


import seaborn as sns

# Sample data
data = heart_data.cigsPerDay 

# Create a distribution plot
sns.histplot(data, color='blue')

# Add labels and title
plt.xlabel('cigsPerDay  Value')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Display the plot
plt.show()


# In[44]:


heart_data['cigsPerDay'].fillna(heart_data['cigsPerDay'].median() , inplace=True)
null_values = heart_data.isnull().sum()
null_values


# In[45]:


count_bpm_zero = (heart_data['blood pressure medication'] == 1).sum()


# In[46]:


count_bpm_zero


# In[47]:


heart_data['blood pressure medication'].fillna(heart_data['blood pressure medication'].median() , inplace=True)


# In[48]:


count_bpm_zero = (heart_data['blood pressure medication'] == 0).sum()


# In[49]:


count_bpm_zero


# In[50]:


null_values = heart_data.isnull().sum()
null_values


# In[51]:


heart_data["cholesterol level"].fillna(heart_data['cholesterol level'].mean(),inplace=True)


# In[52]:


null_values = heart_data.isnull().sum()
null_values


# In[53]:


bmi = heart_data.BMI                  

# Create a distribution plot
sns.histplot(bmi, color='blue')

# Add labels and title
plt.xlabel('BMI  Value')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Display the plot
plt.show()


# In[54]:


heart_data["BMI"].fillna(heart_data['BMI'].mean(),inplace=True)
null_values = heart_data.isnull().sum()
null_values


# In[55]:


heartRate = heart_data.heartRate                              

# Create a distribution plot
sns.histplot(heartRate, color='blue')

# Add labels and title
plt.xlabel('heartRate')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Display the plot
plt.show()


# In[56]:


heart_data["heartRate"].fillna(heart_data['heartRate'].mean(),inplace=True)
null_values = heart_data.isnull().sum()
null_values


# In[57]:


glucose= heart_data.glucose                                          

# Create a distribution plot
sns.histplot(glucose, color='blue')

# Add labels and title
plt.xlabel('heartRate')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Display the plot
plt.show()


# In[58]:


heart_data["glucose"].fillna(heart_data['glucose'].mean(),inplace=True)
null_values = heart_data.isnull().sum()
null_values


# In[59]:


heart_data


# ## Model Building

# In[60]:


X=heart_data.iloc[:,:-1].values
y=heart_data.iloc[:,-1].values


# In[88]:


# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[105]:


model.fit(X_train,y_train)


# In[106]:


y_pred=model.predict(X_test)


# In[107]:


y_pred


# In[108]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# #### Support Vector Machine

# In[105]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report


# In[117]:


m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train,y_train)


# In[119]:


y_prediction=svc.predict(X_test)


# In[120]:


y_prediction


# In[121]:


from sklearn.metrics import confusion_matrix, accuracy_score
confm = confusion_matrix(y_test, y_prediction)
print(confm)
accuracy_score(y_test, y_prediction)


# In[74]:




