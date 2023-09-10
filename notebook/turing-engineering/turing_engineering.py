#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
import numpy as np
from typing import List
from pathlib import Path
from pandas import DataFrame


# # 数据读取

# In[2]:


data_source = Path('data').joinpath('data.csv')
data_source


# In[3]:


df = pd.read_csv(data_source)
df.head()


# # 数据预处理

# ### 变量类型判断

# In[4]:


# 根据unique值数量初筛连续变量和分类变量,阈值可调整
target_col = '是否私行客户'
_cols = set(df.columns.to_list())
_cols.remove(target_col)

num_col_threshold = 20 # 超过20个不同变量则为Numerical
num_cols = []
cate_cols = []
for col in _cols:
    unique_vals = list(df[col].unique())
    if len(unique_vals) > num_col_threshold:
        num_cols.append(col)
    else:
        cate_cols.append(col)


# In[5]:


num_cols


# In[6]:


cate_cols


# ### 变量类型校准

# In[7]:


num_cols.remove('date')


# ## 数据处理

# ### 删除空值比例高于X%的列

# In[8]:


def remove_missing_cols(df:DataFrame, threshold:float = 0.8):
    df = df.copy()
    _drop_cols = []
    for col in df.columns: 
        percent_missing = df[col].isnull().sum() / len(df)
        if percent_missing > threshold:
            _drop_cols.append(col)
    return df.drop(_drop_cols, axis=1)


# In[9]:


df_processed = remove_missing_cols(df,0.8)
df_processed.head()


# ### 缺失值统计

# In[10]:


def null_statistic(df):
    return df.isnull().sum().to_csv('null_statistic.csv')


# In[11]:


null_statistic(df_processed)


# ### 缺失值填充

# In[12]:


def null_filler(series: pd.Series, option: str, quantile_val:float = None) -> pd.Series :
    if pd.api.types.is_string_dtype(series):
        return series.mode()
    
    if option == 'mean':
        return series.mean()
    
    if option == 'median':
        return series.median()
    
    if option == 'mode':
        return series.mode()
    
    if option == 'quantile':
        return series.quantile(quantile_val)
    
    
    return 0
    

def fill_null(df:DataFrame, options:str | List[str], quantile_val:float=None) -> DataFrame:
    """Fill dataset with different options
    options can be median,mean,mode, quantile
    """
    _df = df.copy()
    if type(options) == str:
        options = [options] * len(_df.columns)
        
    for col,option in zip(_df.columns,options):
        _val = null_filler(df[col],option, quantile_val)
        _df[col] = _df[col].fillna(_val)
        
    return _df


# In[13]:


df_processed = fill_null(df_processed,'quantile',0.25)
df_processed.head()


# ### 处理异常值

# In[30]:


df_describe = df_processed[num_cols]
df_describe = df_processed.describe()
df_describe


# In[ ]:


# for col_name, col_info in df_describe.to_dict().items():
    # pass
    # df_processed = df_processed[df_processed[col_name] < col_info['75%']]
    # df_processed = df_processed[df_processed[col_name] > col_info['25%']]
df_processed.head()


# ### Dummy variables

# In[15]:


def process_dummies(df:DataFrame,dummy_cols:List) -> DataFrame:
    """Convert categorical variable into dummies"""
    return pd.get_dummies(df,columns=dummy_cols)


# In[16]:


df_processed = process_dummies(df_processed,cate_cols)
df_processed.head()


# ### Add quntile info

# In[17]:


def add_quantiles(df:DataFrame, numeric_cols:List, num_qutiles:int) -> DataFrame:
    """Add quntile info for selecte numeric columns"""
    _df = df.copy()
    for col in numeric_cols:
        _df[f'q_{col}'] = pd.qcut(_df[col],num_qutiles,labels=False)
    return _df


# In[18]:


add_quantiles(df_processed,num_cols,10).head()


# ## Data Analysis

# In[19]:


df_processed.info()


# ## Compose Dataset

# In[20]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[21]:


df_data = df_processed.copy()
target_col = '是否私行客户' # Class column
feature_cols = df_data.columns.to_list() # Features
feature_cols.remove(target_col)
feature_cols.remove('date')


# In[22]:


df_data.head()


# ### Correlation Coefficient

# In[32]:


df_data.corrwith(df[target_col],method='pearson') # Correlation coefficient for eacth feature column


# ### WOE & IV Value

# ## Build models

# In[24]:


df_data = df_data.dropna()
X_train, X_test, y_train, y_test = train_test_split(df_data[feature_cols], df_data[target_col], test_size=0.33, random_state=42)


# ### Normalization

# In[25]:


from sklearn.preprocessing import StandardScaler

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)


# ### Train models

# In[26]:


from sklearn.linear_model import LogisticRegression
reg_log = LogisticRegression()
reg_log.fit(X_train, y_train)
predictions = reg_log.predict(X_test)


# In[27]:


models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()


# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy, precision, recall, f1, TN, FP, FN, TP = {}, {}, {}, {}, {}, {}, {}, {}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, y_train)
    
    # Make predictions
    predictions = models[key].predict(X_test)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    f1[key] = f1_score(predictions,y_test)
    TN[key], FP[key], FN[key], TP[key] = confusion_matrix(y_test, predictions).ravel()


# In[29]:


import pandas as pd

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall','F1'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['F1'] = f1.values()
df_model['TN'] = TN.values()
df_model['FP'] = FP.values()
df_model['FN'] = FN.values()
df_model['TP'] = TP.values()


df_model

