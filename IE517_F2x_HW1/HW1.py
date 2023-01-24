#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn import datasets 
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


# In[3]:


iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

print(X_iris.shape, y_iris.shape)
print(X_iris[0], y_iris[0])


# In[4]:


X, y = X_iris[:, :2], y_iris


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33) 
print(X_train.shape, y_train.shape)


# In[6]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c =colors[i])
plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")


# In[8]:


clf = SGDClassifier()
clf.fit(X_train, y_train)


# In[9]:


print(clf.coef_)
print(clf.intercept_)


# In[10]:


x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5


# In[11]:


xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    #error here need plt.
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys)


# In[12]:


print( clf.predict(scaler.transform([[4.7, 3.1]])) )
print( clf.decision_function(scaler.transform([[4.7, 3.1]])) )


# In[13]:


y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )

print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )
print( metrics.confusion_matrix(y_test, y_pred) )


# In[14]:


print("My name is Yu-Ching Liao")
print("My NetID is: 656724372")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




