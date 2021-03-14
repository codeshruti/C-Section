
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


# Load Dataset
cs = pd.read_csv('caesarian.csv')


# In[6]:


from sklearn.model_selection import train_test_split
X = cs.drop(['Caesarian Section'],axis=1)
Y = cs.iloc[:,-1].values


# In[39]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
  


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# In[44]:

clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)


# In[58]:


y_pred = clf_nb.predict(X_test)

# In[69]:


# Trying Streamlit
import joblib
with open('model-v2.joblib', 'wb') as f:
    joblib.dump(clf_nb,f)


# In[70]:


def yes_or_no(value):
    if value == 'Yes':
        return 1
    else:
        return 0


# In[71]:


def correction(value):
    if value == 'Late' or value == 'Very Late':
        return 2
    elif value == 'Normal':
        return 1
    else:
        return 0


# In[72]:


def correct(value):
    if value == 'Slightly High' or value == 'High':
        return 1
    else:
        return 0


# In[73]:


def get_user_input():
    st.title('C-Section Chance Prediction')
    st.subheader('Momma Parameters')
    Age = st.slider('Your Age',17,49)
    DN = st.selectbox("Enter Delivery Number ",('1','2','3','4'))
    HP_cat = st.selectbox("Do you have heart problem ? ",('No','Yes'))
    HP = yes_or_no(HP_cat)
    DT_cat = st.selectbox("Enter Delivery Time ",('Normal','Premature','Late','Very Late'))
    DT = correction(DT_cat)
    BP_cat = st.selectbox("Enter Blood Pressure status ",('Low','Normal','Slightly High','High'))
    BP = correct(BP_cat)
    features = {'Age': Age,
            'Delivery Number': DN,
            'Heart Problem': HP,
            'Delivery Time': DT,
            'Blood Pressure': BP
               }
    data = pd.DataFrame(features,index=[0])

    return data


# In[74]:


import streamlit as st
user_input_df = get_user_input()

# In[75]:


def visualize_confidence_level(prediction_proba):

    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index=['Less','High'])
    max_percentage = grad_percentage['Percentage'].max()
    min_percentage = grad_percentage['Percentage'].min()
    result_index = grad_percentage.idxmax(axis = 0) 
    if result_index[0] =='Less':
    	st.title(str(min_percentage) + ' % chance of C - Section')
    else:
    	st.title(str(max_percentage) + ' % chance of C - Section')
    return


# In[76]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[77]:


prediction_proba = clf_nb.predict_proba(user_input_df)
visualize_confidence_level(prediction_proba)

