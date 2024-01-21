#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Heart Desease Prediction


# In[1]:


import pandas as pd


# In[4]:


df = pd.read_csv("C:\\Users\\MonishaAnthony\\Downloads\\archive (3)\\heart.csv")


# In[5]:


df


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[9]:


df.shape


# In[23]:


##Removing duplicate datas


# In[24]:


data_dup = df.duplicated().any()


# In[25]:


data_dup


# In[26]:


df= df.drop_duplicates()


# In[27]:


data_dup = df.duplicated().any()


# In[28]:


data_dup


# In[ ]:


##Serperating categorical and numerical column for data processing


# In[61]:


cate_colum=[]
cont_colum=[]

for column in df.columns:
    if df[column].nunique()<=8:
        cate_colum.append(column)
    else:
        cont_colum.append(column)


# In[62]:


cate_colum


# In[63]:


cont_colum


# In[64]:


df.head()


# In[65]:


df['thal'].unique()


# In[66]:


cate_colum.remove('sex')
cate_colum.remove('fbs')
cate_colum.remove('exang')
cate_colum.remove('target')


# In[68]:


cate_colum


# In[69]:


## putting dummy datas for categorical colum which has more than 2 categories


# In[70]:


df= pd.get_dummies(df,columns=cate_colum,drop_first=True)


# In[71]:


df.head()


# In[ ]:


## Feature scaling


# In[72]:


from sklearn.preprocessing import StandardScaler


# In[75]:


st= StandardScaler()
df[cont_colum]= st.fit_transform(df[cont_colum])


# In[77]:


df.head()


# In[78]:


##Spliting the data into training and testing


# In[81]:


X= df.drop('target', axis = 1)


# In[82]:


X


# In[85]:


Y= df['target']


# In[86]:


Y


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train,x_test,Y_train,y_test= train_test_split(X,Y, test_size=0.2,random_state=42)


# In[89]:


X_train


# In[91]:


Y_train


# In[92]:


##Logistic Regression


# In[93]:


from sklearn.linear_model import LogisticRegression


# In[94]:


log = LogisticRegression()
log.fit(X_train,Y_train)


# In[96]:


Y_pred =log.predict(x_test)


# In[97]:


from sklearn.metrics import accuracy_score


# In[99]:


accuracy_score(y_test,Y_pred)


# In[ ]:


##Support vector Classifier


# In[100]:


from sklearn import svm


# In[101]:


svm = svm.SVC()


# In[102]:


svm.fit(X_train,Y_train)


# In[103]:


y_pred1= svm.predict(x_test)


# In[105]:


accuracy_score(y_test,y_pred1)


# In[106]:


##K-Neighbors Classifier


# In[107]:


from sklearn.neighbors import KNeighborsClassifier


# In[112]:


knei = KNeighborsClassifier(n_neighbors=2)


# In[113]:


knei.fit(X_train,Y_train)


# In[114]:


Y_pred2 = knei.predict(x_test)


# In[115]:


accuracy_score(y_test,Y_pred2)


# In[117]:


##NON-Linear ML Algoritms


# In[118]:


df = pd.read_csv("C:\\Users\\MonishaAnthony\\Downloads\\archive (3)\\heart.csv")


# In[119]:


df.head()


# In[123]:


df= df.drop_duplicates()


# In[124]:


df.shape


# In[127]:


X= df.drop('target',axis=1)


# In[128]:


X


# In[131]:


Y=df['target']


# In[132]:


Y


# In[133]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[134]:


## Decision tree classifier


# In[135]:


from sklearn.tree import DecisionTreeClassifier


# In[136]:


dtc = DecisionTreeClassifier()


# In[137]:


dtc.fit(X_train,Y_train)


# In[138]:


y_pred3= dtc.predict(X_test)


# In[140]:


accuracy_score(Y_test,y_pred3)


# In[ ]:


##Random Forest Classifier


# In[141]:


from sklearn.ensemble import RandomForestClassifier


# In[142]:


rsc = RandomForestClassifier()


# In[143]:


rsc.fit(X_train,Y_train)


# In[144]:


Y_pred4= rsc.predict(X_test)


# In[145]:


accuracy_score(Y_test,Y_pred4)


# In[146]:


##Gradient Boosting Classifer


# In[147]:


from sklearn.ensemble import GradientBoostingClassifier


# In[148]:


gbc = GradientBoostingClassifier()


# In[149]:


gbc.fit(X_train,Y_train)


# In[150]:


Y_pred5= gbc.predict(X_test)


# In[151]:


accuracy_score(Y_test,Y_pred5)


# In[159]:


final_data = pd.DataFrame({'Models': ['LR','SVC','KNN','DTC','RFC','GBC'],'ACC': [accuracy_score(y_test,Y_pred),accuracy_score(y_test,y_pred1),
                                    accuracy_score(y_test,Y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,Y_pred4),accuracy_score(y_test,Y_pred5)]})


# In[164]:


final_data


# In[177]:


## The Random forest classifier is best for prediction so using Random forest classifier for entire data


# In[178]:


X= df.drop('target',axis=1)


# In[179]:


Y=df['target']


# In[180]:


from sklearn.ensemble import RandomForestClassifier


# In[181]:


rfc_data = RandomForestClassifier()


# In[182]:


rfc_data.fit(X,Y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[183]:


## Doing sample prediction using new datas


# In[184]:


import pandas as pd


# In[185]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])


# In[186]:


new_data


# In[188]:


p = rfc_data.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




