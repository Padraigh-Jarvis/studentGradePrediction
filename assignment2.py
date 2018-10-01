# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:11:10 2017

@author: Padraigh Jarvis
"""
import timeit
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors


from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


def nominalToNumber(dataframe,column,mapping):
    dataframe[column] = dataframe[column].map(mapping)
    return dataframe



def removeUselessData(dataframe):
     dataframe = dataframe.drop('school',axis=1)
#     dataframe = dataframe.drop('G1',axis=1)
#     dataframe = dataframe.drop('G2',axis=1)
     return dataframe
 
    
def encodeCatData(studentData):
    sexMapping ={'M':0, 'F':1}
    studentData = nominalToNumber(studentData,'sex',sexMapping)
    addressMapping = {'R':0,'U':1}
    studentData = nominalToNumber(studentData,'address',addressMapping)
    
    familySizeMapping = {'GT3':0,'LE3':1}
    studentData = nominalToNumber(studentData,'famsize',familySizeMapping)
    
    pStatusMapping = {'T':0,'A':1}
    studentData = nominalToNumber(studentData,'Pstatus',pStatusMapping)
    
    parentJobMapping = {'at_home':0,'health':1,'other':2,'services':3,'teacher':4}
    studentData = nominalToNumber(studentData,'Mjob',parentJobMapping)
    studentData = nominalToNumber(studentData,'Fjob',parentJobMapping)    
    
    reasonMapping = {'course':0,'other':1,'home':2,'reputation':3}
    studentData = nominalToNumber(studentData,'reason',reasonMapping)
    
    guardianMapping = {'mother':0,'father':1,'other':2}
    studentData = nominalToNumber(studentData ,'guardian' ,guardianMapping)
    
    yesNoMapping = {'yes':0,'no':1}
    studentData = nominalToNumber(studentData,'schoolsup',yesNoMapping)
    studentData = nominalToNumber(studentData,'famsup',yesNoMapping)
    studentData = nominalToNumber(studentData,'paid',yesNoMapping)
    studentData = nominalToNumber(studentData,'activities',yesNoMapping)
    studentData = nominalToNumber(studentData,'nursery',yesNoMapping)
    studentData = nominalToNumber(studentData,'higher',yesNoMapping)
    studentData = nominalToNumber(studentData,'internet',yesNoMapping)
    studentData = nominalToNumber(studentData,'romantic',yesNoMapping)
    return studentData


def assignment2():
    studentData = pd.read_csv('student-mat.csv')
    
    #Remove school feature as all values for school in this dataset are the same
    studentData = removeUselessData(studentData)
    
    #turn nominal and binary string values to numberic
    studentData = encodeCatData(studentData)
    
    
    gradeMapping = {0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:1,10:2
                   ,11:2,12:2,13:2,14:2,15:3,16:3,17:3,18:3,19:3,20:3}
    studentData=nominalToNumber(studentData,'G3',gradeMapping)
    studentData=nominalToNumber(studentData,'G2',gradeMapping)
    studentData=nominalToNumber(studentData,'G1',gradeMapping)
    
    corRes= studentData.corr()
    sns.heatmap(corRes)
    plt.show()
    
    #remove target 
    finalGradeSeries = studentData['G3']
    studentData = studentData.drop('G3',axis=1)
    
    
    #Decision Tree
    clf = DecisionTreeClassifier()
    decisionScores = model_selection.cross_val_score(clf, studentData, finalGradeSeries, cv=10)
    print("Decision tree:",decisionScores.mean())
    
    #Naïve Bayes
    clf = GaussianNB()
    bayesScores = model_selection.cross_val_score(clf , studentData, finalGradeSeries,cv=10) 
    print("Naive Bayes:",bayesScores.mean())
    
    #Nerest neighbor
    clf = neighbors.KNeighborsClassifier()
    knnScores = model_selection.cross_val_score(clf,studentData,finalGradeSeries,cv=10)
    print("Nearest neighbor:",knnScores.mean())
    
    
    #Random Forest
    clf = RandomForestClassifier()
    randomForScore = model_selection.cross_val_score(clf,studentData,finalGradeSeries,cv=10)
    print("Random Forest:",randomForScore.mean())
    
    #Logistical regression
    clf= LogisticRegression()  
    logRegScore = model_selection.cross_val_score(clf,studentData,finalGradeSeries,cv=10)
    print("Logistical regression",logRegScore.mean())

    #SVC
    clf = SVC()
    scores = model_selection.cross_val_score(clf,studentData,finalGradeSeries,cv=10)    
    print("SVC:",scores.mean())
    
    #hyper param
    #SVC
    param_gridSVC = [ {'C':[0.001,0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001, 0.01, 0.1, 1]} ]
    clf = model_selection.GridSearchCV(SVC(), param_gridSVC, cv=10)
    clf.fit(studentData,finalGradeSeries)
    print("Hyper param SVC:",clf.best_params_ ," score ", clf.best_score_)
    
    
    
    #unused HPO code    
#    #Logistical Regression
#    param_gridLR = [{'random_state':[0,1,2,3,4,5,6,7], 'C':[0.001,0.01,0.1,1,10,100,1000,10000] } ]
#    clf = model_selection.GridSearchCV(LogisticRegression(),param_gridLR,cv=10)
#    clf.fit(studentData,finalGradeSeries)
#    print("Hyper param Logistical regression:",clf.best_params_ ," score ", clf.best_score_)
#    
#    #Random Forest    
#    param_gridRF= [{'n_estimators': list(range(10,190,10)), 'criterion':["gini","entropy"] ,'max_features' :["auto", "log2", "sqrt"]  }]
#    clf = model_selection.GridSearchCV(RandomForestClassifier(),param_gridRF,cv=10)
#    clf.fit(studentData,finalGradeSeries)
#    print("Hyper param Random Forest:",clf.best_params_ ," score ", clf.best_score_)
#    
#    #Nerest neighbor 
#    param_gridknn = [ {'n_neighbors': list(range(1, 20)), 'p':[1, 2, 3, 4, 5] , 'weights':["uniform", "distance"]} ]
#    clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), param_gridknn, cv=10)
#    clf.fit(studentData,finalGradeSeries)
#    print("Hyper param KNN:",clf.best_params_ ," score ", clf.best_score_)
#
#    #Decision tree
#    param_gridDT= [{'criterion':["gini","entropy"], 'random_state':[0,1,2,3,4,5,6,7]}]
#    clf = model_selection.GridSearchCV(DecisionTreeClassifier(),param_gridDT,cv=10)
#    clf.fit(studentData,finalGradeSeries)
#    print("Hyper param decision tree:",clf.best_params_ ," score ", clf.best_score_)
    

    #confusion matrix code 
    classifier = SVC(**clf.best_params_)
    studentData=studentData.values
    aggregrateCnfMatrix = np.zeros((4,4))
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    
    for train_index, test_index in kf.split(studentData):
        classifier.fit(studentData[train_index], finalGradeSeries[train_index])
        y_pred = classifier.predict(studentData[test_index])
        cnf_matrix = confusion_matrix(finalGradeSeries[test_index], y_pred)
        aggregrateCnfMatrix += cnf_matrix
        
    plot_confusion_matrix(cnf_matrix, classes=['0-24', '25-49' , '50-74' , '75-100'])
    
    #Feature selection
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(studentData, finalGradeSeries)
    importances = forest.feature_importances_
  
    test = np.argsort(importances)
    print(test)

    
    listToRemove=[test[0]]
    accList=[]
    for i in range(0,29):
        trainingCopy = np.delete(studentData,listToRemove,axis=1)
        scores = model_selection.cross_val_score(classifier,trainingCopy,finalGradeSeries,cv=10)
        print("SVC:",scores.mean(),"after feature",test[i],"removed")
        accList.append(scores.mean())
        listToRemove.append(test[i+1])
    
    
    
    plt.plot(range(0,29),accList)
    plt.show()
    
    
    
    
    #Code copyed form the scikit website  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
if __name__ == '__main__':
    start = timeit.default_timer()
    assignment2()
    stop=timeit.default_timer()
    print(stop-start , "seconds taken")
