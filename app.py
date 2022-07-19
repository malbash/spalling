#!/usr/bin/env python
# coding: utf-8

# In[2]:


from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import streamlit as st


# In[3]:


st.write ( """spalling""")


# In[4]:


st.sidebar.header('user input Parameters')


# In[5]:


def user_input_features():
    Maximum_exposure_temperature = st.sidebar.slider('tempooo',100.0,1300.0,500.0)
    Moisture_content = st.sidebar.slider('moisture',0.0,0.07,0.02)
    Water_binder= st.sidebar.slider('Water_binder',0.0,5.3,0.3)
    Aggregate_binder= st.sidebar.slider('Aggregate_binder',0.0,0.3,5.0)
    Sand_binder= st.sidebar.slider('Sand_binder',0.0,0.3,5.0)
    Heating_rate= st.sidebar.slider('Heating_rate',1,5,200)
    Silicafume_binder= st.sidebar.slider('Silicafume_binder',0.0,0.3,5.0)
    Aggregate_size= st.sidebar.slider('Aggregate_size',0.0,0.3,15.0)
    GGBS_binder= st.sidebar.slider('GGBS_binder',0.0,0.3,5.0)
    FA_binder= st.sidebar.slider('FA_binder',0.0,0.3,5.0)
    PPfiber_quantity= st.sidebar.slider('PPfiber_quantity',0.0,0.3,20.0)
    PPfiber_diameter= st.sidebar.slider('PPfiber_diameter',0.0,20.0,100.0)
    PPfiber_length= st.sidebar.slider('PPfiber_length',0.0,20.0,100.0)
    Steelfiber_quantity= st.sidebar.slider('Steelfiber_quantity',0.0,0.3,1000.0)
    Steelfiber_diameter= st.sidebar.slider('Steelfiber_diameter',0.0,20.0,100.0)
    Steelfiber_length= st.sidebar.slider('Steelfiber_length',0.0,20.0,100.0)
    
    data={'Maximum_exposure_temperature': Maximum_exposure_temperature,
          'Moisture_content': Moisture_content,    
          'Water_binder':Water_binder,
          'Aggregate_binder':Aggregate_binder,
          'Sand_binder':Sand_binder,
         'Heating_rate':Heating_rate,
         'Silicafume_binder':Silicafume_binder,
         'Aggregate_size':Aggregate_size,
         'GGBS_binder':GGBS_binder,
         'FA_binder':FA_binder,
         'PPfiber_quantity':PPfiber_quantity,
          'PPfiber_diameter':PPfiber_diameter,
          'PPfiber_length':PPfiber_length,
          'Steelfiber_quantity':Steelfiber_quantity,
          'Steelfiber_diameter':Steelfiber_diameter,
          'Steelfiber_length':Steelfiber_length
         }
    features = pd.DataFrame(data,index=[1])
    return features


# In[11]:


df=user_input_features()
#df=df[:1]


# In[12]:


st.subheader('user input')
st.write(df)


# In[13]:


fire=pd.read_csv('app.csv')
x=fire.drop(['Output'],axis=1)
y=fire['Output']
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=111)


# In[14]:


xgbc=xgb.XGBClassifier(objective ='binary:logistic',missing=1,seed=42,learning_rate = 0.05, max_depth = 3)
xgbc.fit(x,y)#_train,y_train,verbose=True,early_stopping_rounds=50,eval_metric='aucpr',eval_set=[(x_test,y_test)])
prediction = xgbc.predict(df)   


# In[20]:


         
st.subheader('prediction')
st.write([prediction])


# In[ ]:





# In[ ]:




