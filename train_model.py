import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

"""
def logisticRegression():
	'''
	:global: none
	:local: lr, ytestoutput, target_names, joblib
	:return:
	:Used to calculate the weights for the features using logistic regression for the dataset
	'''
	print('logistic regression')
	lr = LogisticRegression(tol = 1e-4, C = 1.0, random_state = 0, )
	lr.fit(xt, yt)
	ytestoutput = lr.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ytestoutput))
	print ('ROC', roc_auc_score(ytt, ytestoutput))
	target_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ytestoutput, target_names = target_names))
	joblib.dump(lr, 'lr_model_save.pkl')
"""


def naiveBayes():
	'''
	:global: none
	:local: clf, ypre, target_names, joblib
	:return: none
	:Used to calculate the weights for the features using naive bayes for the dataset
	'''
	print('naive bayes')
	clf = GaussianNB()
	clf.fit(xt, yt)
	ypre = clf.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ypre))
	print ('ROC', roc_auc_score(ytt, ypre))
	target_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ypre, target_names = target_names))
	print ("")
	joblib.dump(clf, 'GaussNB_model_save.pkl')


"""
def bernoulliNB():
	'''
	:global: none
	:local: clf, ypre, target_names, joblib
	:return: none
	:Used to calculate the weights for the features using bernoulli model for the dataset
	'''
	clf = BernoulliNB()
	clf.fit(xt, yt)
	ypre = clf.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ypre))
	print ('Roc', roc_auc_score(ytt, ypre))
	target_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ypre, target_names = target_names))
	print ("")
	joblib.dump(clf, 'BernoulliNB_model_save.pkl')

def kNN():
	'''
	:global: none
	:local: neigh, ypre, target_names, joblib
	:return: none
	:Used to calculate the weights for the features using Knn model for the dataset
	'''
	print('knn')
	neigh = KNeighborsClassifier(n_neighbors = 5)
	neigh.fit(xt, yt)
	ypre = neigh.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ypre))
	print ('Roc', roc_auc_score(ytt, ypre))
	taret_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ypre, target_names = target_names))
	joblib.dump(neigh, 'KNN_model_save.pkl')
"""

def svm():
	'''
	:global: none
	:local: clf, ypre, target_names, joblib
	:return: none
	:Used to calculate the weights for the features using svm model for the dataset
	'''
	print('svm')
	clf = SVC(gamma='auto')
	clf.fit(xt, yt)
	ypp = clf.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ypp))
	print ('Roc', roc_auc_score(ytt, ypp))
	target_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ypp, target_names = target_names))
	joblib.dump(clf, 'SVM_model_save.pkl')

"""

def randomForest():
	'''
	:global:none
	:local:rf, ytestoutput, target_names
	:return: none
	:Used to calculate the weights for the features using Random Forest algorithm for the dataset
	'''
	print("Random Forest")
	rf = RandomForestClassifier(n_estimators = 250, random_state = 0, max_depth = 11)
	rf.fit(xt, yt)
	ytestoutput = rf.predict(xtt)
	print ('Accuracy', metrics.accuracy_score(ytt, ytestoutput))
	print ('ROC', roc_auc_score(ytt, ytestoutput))
	target_names = ['insummary', 'notinsummary']
	print(classification_report(ytt, ytestoutput, target_names = target_names))
	print('features sorted by their score:')
	print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse = True))



"""




def extratree():
    global featureWeightVector
    print('extratree')
    etc = ExtraTreesClassifier(n_estimators = 900, random_state = 0, criterion = 'entropy', max_depth = 12)
    etc.fit(xt, yt)
    ytestoutput = etc.predict(xtt)
    print ('Accuracy', metrics.accuracy_score(ytt, ytestoutput))
    print ('ROC', roc_auc_score(ytt, ytestoutput))
    target_names = ['insummary', 'notinsummary']
    print(classification_report(ytt, ytestoutput, target_names = target_names))
    print('features sorted by their score:')
    featureWeightVector = sorted(zip(map(lambda x: round(x, 4), etc.feature_importances_), names), reverse = True)
    print (ytestoutput)



featureWeightVector
data = pd.read_csv("C:/Users/Bijoy/Python Code/My Thesis/finalDataset.csv")
x = data.as_matrix(columns = data.columns[1 : 4])
y = data.as_matrix(columns = data.columns[-1 : ])
y = np.squeeze(y);
xt, xtt, yt, ytt = train_test_split(x, y, test_size = 0.2, random_state = 2)
names = ['Topic Feature', 'Tf-Idf', 'Sentence Position Feature']
file = open('C:/Users/Bijoy/Python Code/My Thesis/featureWeightVector.json', 'w+', encoding = 'utf-8')
json.dump(featureWeightVector, file)
file.close()
#logisticRegression()
#naiveBayes()
#bernoulliNB()
#kNN()
#svm()
#randomForest()
#adaboost()
#gradboost()
extratree()
