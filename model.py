import os
import numpy as np
import string
from keras import models, layers, regularizers, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Embedding, Flatten, Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
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
from sklearn.preprocessing import OneHotEncoder,StandardScaler, MinMaxScaler

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.003), activation='relu', input_shape=(117,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.003),activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.002),activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def kfoldvalidation():
    epoch = 20
    train_targets = pd.read_csv('train_targets.csv', delimiter=',', engine='python')
    train_data = pd.read_csv('train_data.csv', delimiter=',', engine='python')
    one_hot = pd.get_dummies(train_targets, prefix= '_')
    train_targets = pd.concat([train_targets,one_hot],axis=1)
    train_targets = train_targets.drop(['Degree of the injury'], axis=1)
    train_data = train_data.drop(['Accident Description'], axis=1) 
    print(train_targets.head(10))
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets, test_size=0.1, random_state=70)

    model = build_model()

    training_history = model.fit(X_train, y_train, epochs=epoch, batch_size=40, verbose=1)
    history_dict = training_history.history

    loss_values = history_dict['loss']
    epochs = range(1, epoch+1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc_values = history_dict['acc']
    plt.plot(epochs, acc_values, 'b', label='Training acc')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.save('accidentmodel.h5')
    
    results = model.evaluate(X_test, y_test, verbose=0)


    print(results)
    #docstring below show the same implementation with crossfold validation
    '''
    neural_network = KerasClassifier(build_fn = build_model, epochs=80, batch_size=50, verbose=1) 
    results = cross_val_score(estimator=neural_network, X=train_data, y=train_targets, cv=6)

    print(results)
    print(results.mean())
    print(results.std())'''

    return results

def embeddingwords():
    maxlen = 1000
    training_samples = 200
    validation_samples = 10000
    max_words = 10000

    train_data = pd.read_csv('Data/accident_data.csv', delimiter=',', engine='python')
    train_data = train_data['Accident Description']

    train_data = train_data.str.replace('([\d]+)\.','')
    train_data = train_data.str.lstrip()
    
    print(train_data[0:5])
    train_targets = pd.read_csv('train_targets.csv', delimiter= ',', engine='python')
    one_hot = pd.get_dummies(train_targets, prefix='Degree')
    train_targets = pd.concat([train_targets, one_hot], axis=1)
    train_targets = train_targets.drop(['Degree of the injury'], axis=1)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    train_data = pad_sequences(sequences, maxlen=maxlen)
    
    print('Shape of data tensor:', train_data.shape)
    #train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=max_len, padding='post')
    model = Sequential()
    model.add(Embedding(10000, 8, input_length = maxlen))

    model.add(Flatten())
    model.add(Dense(36, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(Dense(16, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(4,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_data,train_targets, epochs=20, batch_size=60, validation_split=0.1)


kfoldvalidation()
#embeddingwords()
