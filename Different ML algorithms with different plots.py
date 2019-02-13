#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import get_dummies
import plotly.graph_objs as go
from sklearn import datasets
import plotly.plotly as py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


# # Version

# In[2]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# Set up for better code readability

# In[4]:


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')
np.random.seed(1234)
get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis

# In[5]:


os.chdir('D://python using jupyter')


# In[6]:


dataset = pd.read_csv('Iris.csv')
iris_Species = pd.read_csv('Iris.csv')


# In[7]:


type(dataset)


# In[8]:


type(iris_Species)


# 1. Scatter plot

# In[9]:


sns.FacetGrid(dataset,hue='Species',size=5)     .map(plt.scatter,'SepalLengthCm','SepalWidthCm')     .add_legend()
plt.show()


# 2. Box plot

# In[11]:


dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.figure()


# In[12]:


sns.boxplot(x='Species', y ='PetalLengthCm',data=dataset)
plt.show()


# In[14]:


ax = sns.boxplot(x='Species', y ='PetalLengthCm',data=dataset)
ax = sns.stripplot(x='Species',y='PetalLengthCm',data=dataset,jitter=True,edgecolor='gray')
plt.show()


# In[15]:


ax = sns.boxplot(x='Species', y ='PetalLengthCm',data=dataset)
ax = sns.stripplot(x='Species',y='PetalLengthCm',data=dataset,jitter=True,edgecolor='gray')

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree = ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')
plt.show()


# 3. Histogram

# In[16]:


dataset.hist(figsize=(15,20))
plt.show()


# In[17]:


dataset['PetalLengthCm'].hist()


# 4. Multivariate plots

# In[18]:


pd.plotting.scatter_matrix(dataset,figsize=(10,10))
plt.show()


# 5. violin plots

# In[19]:


sns.violinplot(data=dataset,x='Species',y='PetalLengthCm')


# 6. Pair plot

# In[20]:


sns.pairplot(data=dataset,hue='Species')


# In[21]:


# updating the diagonal elements in a pairplot to show a kde
sns.pairplot(dataset,hue='Species',diag_kind='kde')


# 7. kde plot

# In[22]:


sns.FacetGrid(dataset,hue='Species',size=5).map(sns.kdeplot,'PetalLengthCm').add_legend()
plt.show()


# 8. joint plot

# In[24]:


sns.jointplot(x='SepalLengthCm', y ='SepalWidthCm',data=dataset,size=10,ratio=10,kind='hex',color='green')
plt.show()


# 9. andrew curves

# In[25]:


from pandas.tools.plotting import andrews_curves
andrews_curves(dataset.drop('Id',axis=1),'Species',colormap='rainbow')
plt.show()


# In[26]:


sns.jointplot(x='SepalLengthCm', y='SepalWidthCm',data=dataset,size=6,kind='kde',color='#800000', space=0)


# 10. heat map

# In[27]:


plt.figure(figsize=(7,4))
sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix_r')
plt.show()


# 11. radviz plot

# In[28]:


from pandas.tools.plotting import radviz
radviz(dataset.drop('Id',axis=1),'Species')


# 12. box plot

# In[29]:


dataset['Species'].value_counts().plot(kind='bar')


# 13. vizualization with plotly

# In[31]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
trace = go.Scatter(x=X[:, 0],
                   y=X[:, 1],
                   mode='markers',
                   marker=dict(color=np.random.randn(150),
                               size=10,
                               colorscale='Viridis',
                               showscale=False))

layout = go.Layout(title='Training Points',
                   xaxis=dict(title='Sepal length',
                            showgrid=False),
                   yaxis=dict(title='Sepal width',
                            showgrid=False),
                  )
 
fig = go.Figure(data=[trace], layout=layout)


# In[32]:


py.iplot(fig)


# # Data Preprocessing

# In[33]:


print(dataset.shape)


# In[34]:


dataset.size # columns * rows


# In[35]:


dataset.isnull().sum() # to check missing values


# In[36]:


print(dataset.info())


# In[37]:


dataset['Species'].unique() # to check the unique items in Species column


# In[38]:


dataset['Species'].value_counts()


# In[39]:


dataset.head(4) # check first 4 rows


# In[40]:


dataset.tail(4) # to chek last 4 rows


# In[41]:


dataset.sample(5) # to get 5 random rows from dataset


# In[42]:


dataset.describe() # to give a statistical summary


# In[43]:


dataset.groupby('Species').count()


# In[44]:


dataset.columns # to check no of columns in dataset


# In[45]:


dataset.where(dataset['Species']=='Iris-setosa')


# In[46]:


dataset[dataset['SepalLengthCm'] > 7.2]


# In[47]:


# Separating the models
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# # Data Cleaning

# In[48]:


cols = dataset.columns
features = cols[0:4]
labels = cols[4]
print(features)
print(labels)


# In[49]:


data_norm = pd.DataFrame(dataset)

for feature in features:
    dataset[feature] = (dataset[feature] - dataset[feature].mean())/dataset[feature].std()
    
print('Averages')
print(dataset.mean())

print('\n Deviation')
print(pow(dataset.std(),2))


# In[50]:


# Shuffle the data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
print(indices)


# In[51]:


# One hot encoding on dataframe
from sklearn.model_selection import train_test_split
y = get_dummies(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# converting to np.arrays so that we can use with tensorflows

X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)


# In[52]:


print(X_train.shape,y_train.shape)


# In[53]:


print(X_test.shape,y_test.shape)


# # Prepare Features and Targets

# In[59]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Applying different Machine learning algorithms

# 1. K-nearest neighbour

# In[62]:


from sklearn.neighbors import KNeighborsClassifier

Model = KNeighborsClassifier(n_neighbors=8)
Model.fit(X_train,y_train)

y_pred = Model.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


print('accuracy score is',accuracy_score(y_pred,y_test))


# 2. Radious neighbour classifier

# In[64]:


from sklearn.neighbors import RadiusNeighborsClassifier
Model = RadiusNeighborsClassifier(radius=8.0)
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 3. Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
Model = LogisticRegression()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 4. Passive aggressive classifier

# In[66]:


from sklearn.linear_model import PassiveAggressiveClassifier
Model  = PassiveAggressiveClassifier()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 5. Naive Bayes

# In[67]:


from sklearn.naive_bayes import GaussianNB
Model = GaussianNB()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 6. Bernoulli Naive Bayes

# In[68]:


from sklearn.naive_bayes import BernoulliNB
Model = BernoulliNB()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 7. Support vector machine

# In[69]:


from sklearn.svm import SVC
Model = SVC()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 8. Nu Support vector classifier

# In[70]:


# it is similar to SVM
from sklearn.svm import NuSVC
Model = NuSVC()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 9. Linear Support Vector Classifier

# In[71]:


from sklearn.svm import LinearSVC
Model = LinearSVC()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 10. decision tree

# In[72]:


from sklearn.tree import DecisionTreeClassifier
Model = DecisionTreeClassifier()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 11. Extra tree classifier

# In[73]:


from sklearn.tree import ExtraTreeClassifier
Model = ExtraTreeClassifier()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_pred,y_test))


# 12. neural network

# In[74]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[75]:


w = np.array([0.2,0.3,0.8])
b= 0.5
x = np.array([0.5, 0.6, 0.1])


# In[76]:


z = w.dot(x)+b
print("z:", z)
print("a:", sigmoid(z))


# In[77]:


# The XOR problem
def activation(z):
    if z > 0:
        return 1
    return 0


# In[78]:


# For AND we could implement a perceptron as:

w = np.array([1, 1])
b = -1
x = np.array([0, 0])
print("0 AND 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 AND 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 AND 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 AND 1:", activation(w.dot(x) + b))


# In[79]:


# For OR we could implement a perceptron as:
w = np.array([1, 1])
b = 0
x = np.array([0, 0])
print("0 OR 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 OR 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 OR 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 OR 1:", activation(w.dot(x) + b))


# In[80]:


from sklearn.neural_network import MLPClassifier
Model = MLPClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
# Summary of the predictions
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 13. random forest

# In[81]:


from sklearn.ensemble import RandomForestClassifier
Model = RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 14. Bagging classifier

# In[82]:


from sklearn.ensemble import BaggingClassifier
Model = BaggingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 15. Adaboost classifier

# In[85]:



from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 16.Gradient Boosting Classifier

# In[86]:


from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 17.  Linear Discriminant Analysis

# In[87]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model=LinearDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 18. Quadratic Discriminant Analysis

# In[88]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Model=QuadraticDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# 19. kmeans

# In[89]:


from sklearn.cluster import KMeans
iris_SP = dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# k-means cluster analysis for 1-15 clusters                                              
from scipy.spatial.distance import cdist
clusters=range(1,15)
meandist=[]

# loop through each cluster and fit the model to the train set
# generate the predicted cluster assingment and append the mean 
# distance my taking the sum divided by the shape
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(iris_SP)
    clusassign=model.predict(iris_SP)
    meandist.append(sum(np.min(cdist(iris_SP, model.cluster_centers_, 'euclidean'), axis=1))
    / iris_SP.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method') 
# pick the fewest number of clusters that reduces the average distance
# If you observe after 3 we can see graph is almost linear


# 20. Back propagation

# In[90]:


def sigmoid(z):
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


# In[91]:


def relu(z):
    """The ReLU activation function."""
    return max(0, z)


# In[92]:


def sigmoid_prime(z):
    """The derivative of sigmoid for z."""
    return z * (1 - z)


# In[93]:


X = np.array([ [ 0, 0, 1 ],
               [ 0, 1, 1 ],
               [ 1, 0, 1 ],
               [ 1, 1, 1 ] ])
print(X)


# In[94]:


y = np.array([[0,0,1,1]]).T
print(y)


# In[95]:


np.random.seed(1)


# In[96]:


n_inputs = 3
n_outputs = 1
#Wo = 2 * np.random.random( (n_inputs, n_outputs) ) - 1
Wo = np.random.random( (n_inputs, n_outputs) ) * np.sqrt(2.0/n_inputs)
print(Wo)


# In[97]:


for n in range(10000):
    # forward propagation
    l1 = sigmoid(np.dot(X, Wo))
    
    # compute the loss
    l1_error = y - l1
    #print("l1_error:\n", l1_error)
    
    # multiply the loss by the slope of the sigmoid at l1
    l1_delta = l1_error * sigmoid_prime(l1)
    #print("l1_delta:\n", l1_delta)
    
    #print("error:", l1_error, "\nderivative:", sigmoid(l1, True), "\ndelta:", l1_delta, "\n", "-"*10, "\n")
    # update weights
    Wo += np.dot(X.T, l1_delta)

print("l1:\n", l1)


# 21. More complex example with back progation

# In[98]:


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print(X)


# In[99]:


y = np.array([[ 0, 1, 1, 0]]).T
print(y)


# In[100]:


np.random.seed(1)


# In[101]:


n_inputs = 3
n_hidden_neurons = 4
n_output_neurons = 1
Wh = np.random.random( (n_inputs, n_hidden_neurons) )  * np.sqrt(2.0/n_inputs)
Wo = np.random.random( (n_hidden_neurons, n_output_neurons) )  * np.sqrt(2.0/n_hidden_neurons)
print("Wh:\n", Wh)
print("Wo:\n", Wo)


# In[102]:


for i in range(100000):
    l1 = sigmoid(np.dot(X, Wh))
    l2 = sigmoid(np.dot(l1, Wo))
    
    l2_error = y - l2
    
    if (i % 10000) == 0:
        print("Error:", np.mean(np.abs(l2_error)))
    
    # gradient, changing towards the target value
    l2_delta = l2_error * sigmoid_prime(l2)
    
    # compute the l1 contribution by value to the l2 error, given the output weights
    l1_error = l2_delta.dot(Wo.T)
    
    # direction of the l1 target:
    # in what direction is the target l1?
    l1_delta = l1_error * sigmoid_prime(l1)
    
    Wo += np.dot(l1.T, l2_delta)
    Wh += np.dot(X.T, l1_delta)

print("Wo:\n", Wo)
print("Wh:\n", Wh)


# In[103]:


from sklearn import datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target


# In[104]:


plt.figure('sepal')
colormarkers = [ ['red','s'], ['greenyellow','o'], ['blue','x']]
for i in range(len(colormarkers)):
    px = X_iris[:, 0][y_iris == i]
    py = X_iris[:, 1][y_iris == i]
    plt.scatter(px, py, c=colormarkers[i][0], marker=colormarkers[i][1])

plt.title('Iris Dataset: Sepal width vs sepal length')
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.figure('petal')

for i in range(len(colormarkers)):
    px = X_iris[:, 2][y_iris == i]
    py = X_iris[:, 3][y_iris == i]
    plt.scatter(px, py, c=colormarkers[i][0], marker=colormarkers[i][1])

plt.title('Iris Dataset: petal width vs petal length')
plt.legend(iris.target_names)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


# In[ ]:




