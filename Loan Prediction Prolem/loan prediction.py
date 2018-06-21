
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
X = train.iloc[:,0:11].values
Y = train.iloc[:,11].values
X_test = test.iloc[:,0:11].values

#### WHEN YOU USE THE ORIGINAL DATASET.........
# APPLYING LABEL ENCODER AND ONE HOT ENCODER TO TRAINING DATASET
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()

#X[: ,0] = labelencoder_X.fit_transform(X[: ,0].astype(str))
#X[: ,1] = labelencoder_X.fit_transform(X[: ,1].astype(str))
#X[: ,3] = labelencoder_X.fit_transform(X[: ,3].astype(str))
#X[: ,4] = labelencoder_X.fit_transform(X[: ,4].astype(str))
#X[: ,10] = labelencoder_X.fit_transform(X[: ,10].astype(str))

#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()

# APPLYING LABEL ENCODER AND ONEHOTENCODER TO TEST DATASET
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_test = LabelEncoder()

#X_test[: ,0] = labelencoder_X_test.fit_transform(X_test[: ,0].astype(str))
#X_test[: ,1] = labelencoder_X_test.fit_transform(X_test[: ,1].astype(str))
#X_test[: ,3] = labelencoder_X_test.fit_transform(X_test[: ,3].astype(str))
#X_test[: ,4] = labelencoder_X_test.fit_transform(X_test[: ,4].astype(str))
#X_test[: ,10] = labelencoder_X_test.fit_transform(X_test[: ,10].astype(str))


#onehotencoder = OneHotEncoder()
#X_test = onehotencoder.fit_transform(X_test).toarray()

#DATA VISUALISATION
train.plot(kind='box', subplots=True,  sharex=False, sharey=False)
plt.show()

# histograms
train.hist()
plt.show()


#compare all the known algos

from sklearn import model_selection
# Spot Check Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcess
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVR',SVR()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('GP', GaussianNB()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold,)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X,Y)

# PREDICTING THE DATA
Y_pred = classifier.predict(X_test)
np.reshape(-1,1)
Y_pred.fit_transform(X_test)

#COPYING PREDICTED DATA TO CSV FILE
prediction = pd.DataFrame(Y_pred, columns=['Y_pred']).to_csv('prediction.csv')


