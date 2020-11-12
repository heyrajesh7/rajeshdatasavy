#!/usr/bin/env python
# coding: utf-8

# # AUTHOR - RAJESH KUMAR S

# ## TSF GRIP TASK-4

# ## Question Statement - Prediction using Decision Tree Algorithm

# ## Dataset given - https://bit.ly/3kXTdox

# ## Importing required libraries

# In[23]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import pydotplus
from sklearn import tree
from sklearn.tree import export_graphviz
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


# ## Loading Datasets

# In[24]:


iris=datasets.load_iris()


# In[25]:


d=pd.DataFrame(iris.data, columns=iris.feature_names)
print(d.head(5))

y=iris.target
print(y)


# In[26]:


d.head()


# In[8]:


d.info()
d.describe()


# ## Check Null Values (Data Cleansing)

# In[9]:


d.isnull().sum()


# Note: We can see all the null values and the datatype is int

# ## Data Visualization

# In[10]:


import seaborn as sns
iris=sns.load_dataset('iris')


# In[11]:


iris.head()


# In[14]:


sns.set()
sns.pairplot(iris, hue='species', size= 2.5)


# In[15]:


d.corr()


# In[16]:


plt.figure(figsize=(10,4))
sns.heatmap(d.corr(),annot=True,cmap="cubehelix_r")
plt.show()


# ## Create dummy variables

# In[18]:


dummy=pd.get_dummies(iris['species'])
dummy.head()


# ## Data Splitting

# In[28]:


X_train,X_test,y_train,y_test = train_test_split(d,y, random_state=101, stratify=y, test_size=0.2)


# ## Fitting Model

# In[29]:


model=DecisionTreeClassifier()
model


# In[30]:


r=model.fit(X_train,y_train)


# In[31]:


X_train.shape,y_train.shape


# In[32]:


X_test.shape,y_test.shape


# ## ACCURACY 

# In[33]:


model.score(X_test,y_test)


# In[34]:


model.score(X_train,y_train)


# ## Prediction Model

# In[35]:


y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[32]:


model.predict(X_test)


# In[33]:


model.predict(X_train)


# ## By using Decision-Tree method

# In[36]:


tra_acc=[]
tes_acc=[]
for i in range(1,10):
    treer= DecisionTreeClassifier(max_depth=i,random_state=10)
    treer.fit(X_train,y_train)
    tra_acc.append(treer.score(X_train,y_train))
    tes_acc.append(treer.score(X_test,y_test))


# In[43]:


frame = pd.DataFrame({'max_depth':range(1,10),'train_acc':tra_acc, 'valid_acc':tes_acc})
frame.head()


# In[45]:


plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'],frame['train_acc'], marker='o',label='training accuracy')
plt.plot(frame['max_depth'],frame['valid_acc'],marker='o',label='validation accuracy')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend()


# The above graph representation tells you the relation betweer depth and performnce in x and y-axis successfully.

# In[46]:


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


dot_data = StringIO()
export_graphviz(r, out_file=dot_data, feature_names=X_train.columns,
               filled=True, rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# ## Conclusion

# Thus we have made the output and the decision tree has been derived successfully.
