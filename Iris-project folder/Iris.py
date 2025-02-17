
# Check the versions of libraries
 
# Python version
import sys as sys
print('Python: {}'.format(sys.version))
# scipy
import scipy as sc
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib as plt
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn 
print('sklearn: {}'.format(sklearn.__version__))

#Importing Dataset
dataset = pd.read_csv('Iris.csv')
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# shape
print(dataset.shape)
# head
print(dataset.head(20))

# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('Species').size())

#plot
dataset.plot(kind='box', subplots=True,  sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()
from pandas.plotting import scatter_matrix
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

#Making our Model

X = dataset.iloc[:,1:5].values
Y = dataset.iloc[:,5].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, train_size = 0.2, random_state = 7)


from sklearn.metrics import accuracy_score
scoring = 'accuracy'

#compare all the known algos

from sklearn import model_selection
# Spot Check Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))



































