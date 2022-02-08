#!/usr/bin/env python
# coding: utf-8

# # Importing Lib

# In[1]:


import numpy as np
from sklearn.datasets import load_digits


# # Loading Dataset

# In[2]:


data1 = load_digits()


# # Summarizing Data

# In[ ]:


print(data1.data)
print(data1.target)

print(data1.data.shape)
print(data1.images.shape)

dataimglen = len(data1.images)
print(dataimglen)


# # Data Visualization

# In[5]:


n = 7
import matplotlib.pyplot as mp
mp.gray()
mp.matshow(data1.images[n])
mp.show()

data1.images[n]


# # Data Segeregation

# In[6]:


x = data1.images.reshape((dataimglen,-1))
x


# In[7]:


y = data1.target
y


# # Splitting Training and Test data

# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
print(x_train.shape)
print(x_test.shape)


# # Training

# In[9]:


from sklearn import svm
model = svm.SVC(kernel = 'linear')
model.fit(x_train,y_train)


# # Prediction

# In[12]:


n = 7
result = model.predict(data1.images[n].reshape((1,-1)))
mp.imshow(data1.images[n], cmap = mp.cm.gray_r, interpolation = 'nearest')
print(result)
print('\n')
mp.axis('off')
mp.title('%i'%result)
mp.show()


# In[13]:


y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# # Accuracy While using Linear kernel

# In[14]:


from sklearn.metrics import accuracy_score
print("Accuracy of the model :{0}%".format(accuracy_score(y_test,y_pred)*100))


# # Accuracy while using different kernel n gamma values

# In[45]:


from sklearn import svm
m1 = svm.SVC(kernel = 'linear')
m2 = svm.SVC(kernel = 'rbf')
m3 = svm.SVC(gamma = 0.001)
m4 = svm.SVC(gamma = 0.001,C=0.7)

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)

predm1 = m1.predict(x_test)
predm2 = m2.predict(x_test)
predm3 = m3.predict(x_test)
predm4 = m4.predict(x_test)

print("Accuracy of m1 : {0}%".format(accuracy_score(y_test,predm1)*100))
print("Accuracy of m2 : {0}%".format(accuracy_score(y_test,predm2)*100))
print("Accuracy of m3 : {0}%".format(accuracy_score(y_test,predm3)*100))
print("Accuracy of m4 : {0}%".format(accuracy_score(y_test,predm4)*100))


# # End of Module
