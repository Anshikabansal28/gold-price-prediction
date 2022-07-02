#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  



# #  Data collection and processing

# In[2]:


#loading a csv data to a pandas dataframe
gold_data = pd.read_csv("gld_price_data.csv")


# In[4]:


#print first 5 rows in the dataframe
gold_data.head()


# In[5]:


#print last five rows of the dataframe
gold_data.tail()


# In[6]:


#number of rows and columns
gold_data.shape


# In[8]:


#some basic information  about the data
gold_data.info()


# In[9]:


#checking the number of missing values
gold_data.isnull().sum()


# In[10]:


#gettingbthe statistical measures the data
gold_data.describe()


# In[46]:


#find the unique values from categorical features
for col in gold_data.select_dtypes(include ='object').columns:
        print(col)
        print(gold_data[col].unique())


# #### categorical features

# In[48]:


categorical_features=[feature for feature in gold_data.columns if ((gold_data[feature].dtypes=="O") & (feature not in ["GLD"]))]
categorical_features


# ####  Numerical features

# In[51]:


#list of numerical variables
numerical_features = [feature for feature in gold_data.columns if ((gold_data[feature].dtypes !='O') & (feature not in ['GLD']))]
print('number of numerical variables: ',len(numerical_features))
     
#visualise numerical variables
gold_data[numerical_features].head()


# ### discrete numerical features

# In[53]:


discrete_feature=[feature for feature in numerical_features if len(gold_data[feature].unique())<25]
print("Discrete Variable Count: {}".format (len(discrete_feature)))


# #### continuous numerical feature

# In[78]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['GLD']]
print("Continuous feature Count: {}".format (len(continuous_features)))


# #### distribution of continuous numerical feature

# In[79]:


#plot a univariate distribution of continues observations
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features :
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(gold_data[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()


# #### relationship between continuous numerical features and labels

# In[80]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features :
    data=gold_data.copy()
    ax = plt.subplot(12,3,plotnumber)
    plt.scatter(data[feature],data['GLD'])
    plt.xlabel(feature)
    plt.ylabel('GLD')
    plt.title(feature)
    plotnumber+=1
plt.show()


# #### outliers in numerical features 

# In[67]:


#boxplot on numerical features to find outliers
plt.figure(figsize=(20,60), facecolor='pink')
plotnumber =1
for numerical_feature in numerical_features :
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(gold_data[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# #### Explore the correlation between numerical features

# correlation:
# 1.positive corelation
# 2.negative corelation

# In[11]:


correlation= gold_data.corr()


# In[12]:


#constructing a heatmap to understand the correlation
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True,square=True, fmt=' .1f',annot=True,annot_kws={'size':8},cmap='Blues' )


# In[13]:


#correlation values of GLD
print(correlation['GLD'])


# In[14]:


#check the distribution of gold price
sns.distplot(gold_data['GLD'],color='yellow')


# #  splitting the features and target

# In[17]:


X= gold_data.drop(['Date','GLD'],axis=1)
Y= gold_data['GLD']


# In[18]:


print(X)


# In[19]:


#y=containing all gold prices
print(Y)


# # Splitting into traing data and test data
# 

# In[22]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=2)


# In[81]:


len(X_train)


# In[82]:


len(X_test)


# In[83]:


X_train


# # model training

# In[23]:


#random forest algorithm
regressor= RandomForestRegressor(n_estimators=100)


# In[24]:


#training the model
regressor.fit(X_train,Y_train)


# # model evaluation

# In[25]:


#prediction on test data
test_data_prediction = regressor.predict(X_test)


# In[26]:


print(test_data_prediction)


# In[27]:


#R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ",error_score)


# #### compare thebactual values and predicted values in plot

# In[28]:


Y_test = list(Y_test)


# In[55]:


plt.plot(Y_test,color ='blue' , label = 'Actual value')
plt.plot(test_data_prediction,color ='orange' , label ='predicted value')
plt.title('Actual price vs predicted price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

