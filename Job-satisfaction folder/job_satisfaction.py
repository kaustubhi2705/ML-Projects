
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('HR_comma_sep.csv')
X = dataset.iloc[:,0:9].values
Y = dataset.iloc[:,9].values

 # Optional work
 
 # 1.] dimension of dataset
 print(dataset.shape)
 # 2.] peek at the data
 print(dataset.head(20))
 # statistical summary
 print(dataset.describe())
# class distribution
 print(dataset.groupby('left').size())
 
 left = dataset[dataset['left'] == 1]
 
 # DATA VISUALISATION
 #PLOTS
left.plot(kind='box', subplots=True,  sharex=False, sharey=False)
plt.show()
# BAR GRAPHS
plt.rcParams['figure.figsize'] = (20,15)  #THIS IS USED TO ENLARGE THE GRAPHS
left.hist()
plt.show()

# SALES PLOTS
import matplotlib.pyplot as plt
pd.crosstab(left.sales,left.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')

# SALARY PLOTS
import matplotlib.pyplot as plt
pd.crosstab(left.salary,left.left).plot(kind='bar')
plt.title('Salaries comparison of EMPLOYEES')
plt.xlabel('salary')
plt.ylabel('Frequency')
plt.savefig('department_bar_chart')
 
# Scatter plot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# creating test set and train set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.1, random_state = 0)

#building Models
from sklearn import model_selection
# Spot Check Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold )
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


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


# Here we can see the results Random forest has highest accuracy...........
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, random_state = 0)
classifier.fit(X_train,Y_train)

# PREDICTING THE DATA
Y_pred = classifier.predict(X_test)
#print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
#print(classification_report(Y_test, Y_pred))



