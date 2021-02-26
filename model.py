import os
import numpy as np
import string
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
from matplotlib import style
import pandas as pd 
import seaborn as sns
import re
import matplotlib.pyplot as partial_train_data
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split, KFold
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score,precision_recall_curve,roc_curve, accuracy_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler


'''def randomly_forested():
    train_targets = pd.read_csv('train_targets1.csv', delimiter=',', encoding='ascii', engine='python')
    train_data = pd.read_csv('train_data0.csv', delimiter=',', encoding='ascii', engine='python')

    X_train, X_test, y_train, y_test = train_test_split(train_data,train_targets,test_size=0.40,random_state=10)
    random_forest = RandomForestClassifier(n_estimators=250,criterion='gini',min_samples_leaf=1,
    max_features=0.8,oob_score=True,random_state=10,n_jobs=-1)
    #cross_validation
    scores = cross_val_score(random_forest,train_data,train_targets,cv=7,scoring='accuracy')
    print('scores:',scores)
    print('mean:',scores.mean())
    print('STD: ',scores.std())
    
    random_forest.fit(X_train,y_train)

    random_forest.score(X_train,y_train)
    acc_random_forest = round(random_forest.score(X_train,y_train)*100,2)
    test_accuracy = round(random_forest.score(X_test,y_test)*100,2)
        

    print(acc_random_forest)
    print(test_accuracy)
    print('oob_score:',round(random_forest.oob_score_,4)*100,'%')    
    #roc_auc score only used for binary(2) labels,,bit you can binarize labels and use it anyway
    y_scores = random_forest.predict_proba(train_data)
    y_scores = y_scores[:,1]
    #r_a_score = roc_auc_score(train_targets,y_scores)
    #print('roc-auc-score:',r_a_score)


'''

def build_model(data):
    #building the achitecture
    data = data
    model = models.Sequential()
    model.add(layers.Dense(100,activation='relu',input_shape=(data.shape[1],)))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(4))
    #compiling...ie showing the loss function(depending on output), how it will be minimised(optimiser)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    #mse=mean squared error,mae = mean absolute error
    return model

def kfoldvalidation():
    train_targets = pd.read_csv('train_targets1.csv', delimiter=',', encoding='ascii', engine='python')
    train_data = pd.read_csv('train_data0.csv', delimiter=',', encoding='ascii', engine='python')
    one_hot = pd.get_dummies(train_targets, prefix= '_')
    train_targets = pd.concat([train_targets,one_hot],axis=1)
    #train_targets = train_targets.loc[:, train_targets.columns.str.contains('^Degree')]

    print(train_targets.columns.tolist())
    train_targets = train_targets.drop('Degree of the injury', axis=1)
    print(train_targets.head(10))
    
    k = 5
    kf = KFold(n_splits=k, random_state=25)
    model = build_model(train_data)

    result = cross_val_score(model , train_data, train_targets, cv = kf, scoring='accuracy')
    print("Avg accuracy: {}".format(result.mean()))
    
    '''
    k=5
    num_val_samples = len(train_data)//k
    num_epochs = 100
    all_scores = []
    for i in range(k):
        print('processing fold #',i)
        val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
        partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
        
        model = build_model(train_data)
        model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=50,verbose=1)
        
        results = model.evaluate(val_data,val_targets,verbose=0)
        all_scores.append(results)
    print(all_scores)'''
    #return 
    
kfoldvalidation()
#randomly_forested()