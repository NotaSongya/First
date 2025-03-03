#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import joblib
import pickle
import csv
#%matplotlib inline

#* BUERABIBX
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import neighbors

# MBER

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

# DRBAX
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
 


# In[2]:


header = ["id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label"]

data = []
for column in header:
    value = input("Enter {}: ".format(column))
    data.append(value)

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow(data)
    


# In[4]:


#test_data_path = input("Enter test data file path: ")
#test_data = pd.read_csv(test_data_path)
test_data = pd.read_csv('output.csv')
X_test = test_data.drop(axis=1, columns=['attack_cat'])
X_test = X_test.drop(axis=1, columns=['label'])

col_trans = joblib.load('col_trans.joblib')
X_test_transform = col_trans.transform(X_test)
 


# In[5]:


# Load the model from file
with open("rmodel.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make predictions on the data
y_pred_class = loaded_model.predict(X=X_test_transform)
y_pred_score = loaded_model.predict_proba(X=X_test_transform)

print(y_pred_class)
 


# In[ ]:




