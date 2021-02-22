import os
import numpy as np
import string
#from keras import models
#from keras import layers
#from keras.utils import to_categorical
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
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score,precision_recall_curve,roc_curve
from sklearn.preprocessing import OneHotEncoder,StandardScaler


def view_data():
    raw_data = pd.read_csv('newgen2011.csv', delimiter=',', engine='python')
    print(raw_data.columns.tolist())
    
    #for repeated data, and data that is uniform all through eg 'Computed', 'Time' is removed due to input errors
    raw_data = raw_data.drop(['Emirate','City','Road','Computed','Computed2','Streets',
                'Gender of the injured','Fasten seat belt','Year','Month','Report type', 'Day',
                'Inp Age','Date - Month', 'Age Group - Ministry', 'Time','Injured person position',
                'Age Group', 'Weather','Road surface','Nationalities', 'Area','Block','Accident Description',
                'Location.1'], axis=1)
    #raw_data.columns.str.contains("^Unnamed")
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]
    print(raw_data.columns.tolist())
    return raw_data

def categorise_age(data):
    '''categorising age'''
    raw_data = data
    raw_data.loc[raw_data['Age of the injured']<=7,'Age of the injured'] = 0
    raw_data.loc[(raw_data['Age of the injured']>7) & (raw_data['Age of the injured']<=14),'Age of the injured'] = 1
    raw_data.loc[(raw_data['Age of the injured']>14) & (raw_data['Age of the injured']<=20),'Age of the injured'] = 2
    raw_data.loc[(raw_data['Age of the injured']>20) & (raw_data['Age of the injured']<=30),'Age of the injured']= 3
    raw_data.loc[(raw_data['Age of the injured']>30) & (raw_data['Age of the injured']<=40),'Age of the injured'] = 4
    raw_data.loc[(raw_data['Age of the injured']>40) & (raw_data['Age of the injured']<=50),'Age of the injured'] = 5
    raw_data.loc[(raw_data['Age of the injured']>50) & (raw_data['Age of the injured']<=60),'Age of the injured'] = 6
    raw_data.loc[(raw_data['Age of the injured']>60),'Age of the injured'] = 7

    return raw_data

def convert_gender(data):
    
    raw_data = data
    genders = {'M':0,'F':1}
    raw_data['Gender'] = raw_data['Gender'].map(genders)

    return  raw_data

def convert_week(data):                    #convert names to titles....im guessing corelation with gender too
    raw_data = data
    weeks = {'First week':0,'Second week':1,'Third week':2,'Fourth week':3}
    raw_data['Week'] = raw_data['Week'].map(weeks)

    #to get only one date-time per entry
    raw_data['Date_time'] = raw_data['Date'].astype(str)+' '+raw_data['Iac Rep Time']
    #drop the separate date and time
    raw_data = raw_data.drop(['Date','Iac Rep Time'],axis=1)
    #change the format to datetime
    raw_data['Date_time'] = pd.to_datetime(raw_data['Date_time'])
    raw_data['Date_time'] = raw_data['Date_time'].dt.strftime('%Y%m%d%H%M')
    raw_data['Date_time'].str.replace('[{}]'.format(string.punctuation), '')

    days = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    raw_data['Day.1'] = raw_data['Day.1'].map(days)

    print(raw_data["Day.1"].head(10))

    return raw_data

def convert_reasons(data):
    raw_data = data
    raw_data['Reasons']= raw_data['Reasons'].str.extract('([a-zA-Z ]+)',expand=False)
    raw_data['ReasonsNew'] = raw_data['Reasons'].fillna('Misuse of traffic')
    reasons = pd.get_dummies(raw_data['ReasonsNew'], prefix='Reasons_')
    raw_data = pd.concat([raw_data,reasons],axis=1) 
    raw_data = raw_data.drop(['Reasons','ReasonsNew'], axis=1)
    return raw_data

def convert_stations(data):
    raw_data = data
    #print(raw_data['Police Station'].drop_duplicates())

    stations = pd.get_dummies(raw_data['Police Station'], prefix='_')
    raw_data = pd.concat([raw_data,stations],axis=1) 
    raw_data = raw_data.drop('Police Station', axis=1)
    #print(raw_data.head(30))

    return raw_data

def convert_street(data):
    raw_data = data
    #print(raw_data['Street'].unique())

    raw_data['Street'] = raw_data['Street'].astype(str)
    #too many streets, convert to only street and intersection
    Street_Dictionary = {'intersection':'Intersection','Intersection':'Intersection', 'Street':'Street','street':'Street' }
    raw_data['Street']= raw_data['Street'].str.extract('(intersection)',expand=False)
    raw_data['Street_name'] = raw_data['Street'].map(Street_Dictionary).fillna('Street')

    #print(raw_data['Street_name'].head(20))

    Street_dummies = pd.get_dummies(raw_data['Street_name'],prefix='Streetname')
    raw_data = pd.concat([raw_data,Street_dummies],axis=1)
  
    raw_data = raw_data.drop(['Street','Street_name'],axis=1)

    #print(raw_data.head(30))

    return raw_data

def convert_place(data):
    raw_data = data
    places = pd.get_dummies(raw_data['Place'], prefix='place_')
    raw_data = pd.concat([raw_data,places],axis=1) 
    raw_data = raw_data.drop('Place', axis=1)

    #print(raw_data.head(30))
    return raw_data

def convert_location(data):
    raw_data = data
    locations = pd.get_dummies(raw_data['Location'], prefix='location_')
    raw_data = pd.concat([raw_data,locations],axis=1) 
    raw_data = raw_data.drop('Location', axis=1)

    return raw_data

def convert_lighting(data):
    raw_data = data
    types = {'day':0,'night - enough lights':1}
    raw_data['lighting'] = raw_data['lighting'].map(types)
    print(raw_data.head(10))

    return  raw_data

def convert_intersection(data):
    raw_data = data
    intersections = pd.get_dummies(raw_data['Intersection'], prefix='intersection_')
    raw_data = pd.concat([raw_data,intersections],axis=1) 
    raw_data = raw_data.drop(['Intersection','intersection__????? ??? ?????'], axis=1)

    return raw_data

def convert_accident_type(data):
    raw_data = data
    types = pd.get_dummies(raw_data['Accident Type'], prefix='accidentype_')
    raw_data = pd.concat([raw_data,types],axis=1) 
    raw_data = raw_data.drop('Accident Type', axis=1)

    print(raw_data.head(10))

    return raw_data

def convert_seatbelt(data):
    raw_data = data
    types = {'Y':0,'N':1}
    raw_data['Seat Belt'] = raw_data['Seat Belt'].map(types)
    #print(raw_data['Seat Belt'].head(10))
    #print(raw_data.columns.tolist())
    return raw_data

def convert_injured_person_position(data):
    raw_data = data
    #print(raw_data["Injured person's seat"].drop_duplicates())
    types = {'Front passenger seat':0,'Back passenger seat':1,"Driver's seat":2,'No passenger':3,'bike passenger':4}
    raw_data["Injured person's seat"] = raw_data["Injured person's seat"].map(types)

    #print(raw_data["Injured person's seat"].head(10))

    return raw_data

def convert_nationality_injured_person(data):
    raw_data = data
    types = {'Arab States':0,'Asian States':1,'UAE':2,'GCC':3,'Other':4}
    raw_data["Nationality of the Injured person"] = raw_data["Nationality of the Injured person"].map(types)
    #print(raw_data["Nationality of the Injured person"].head(20))

    
    #print(raw_data['Pedestrian action'].drop_duplicates())
    actions = {'None':0, 'Cross but not on the crosswalk ':1,'Cross on crosswalk not in the intersection':2,
                'Cross on crosswalkS not in the intersection':2,"Cross, there's no crosswalk":5, 'Stop':7, 'Cross without attention':8,
                'Coss, but not on the crosswalk':1, 'Coss on the crosswalk of the intersection':3,
                'Cross on crosswalk of the intersection':3, 'Cross on crosswalk, not in the intersection':4,
                'Cross on the crosswalk of the intersection':3, 'Stop on the road strip':6}
    raw_data['Pedestrian action'] = raw_data['Pedestrian action'].map(actions).fillna(2)

    #print(raw_data['Pedestrian action'].head(30))

    raw_data['Number of Lanes']=raw_data['Number of Lanes'].fillna(3)


    print(raw_data['Number of Lanes'].head(170))

    raw_data.to_csv('SAMPLE8.csv',index=False)
    print(raw_data.shape)

    return raw_data







clean = view_data()
agelove = categorise_age(clean)
weeklove = convert_gender(agelove)
reasonlove = convert_week(weeklove)
stationlove = convert_reasons(reasonlove)
streetlove = convert_stations(stationlove)
placelove = convert_street(streetlove)
locationlove = convert_place(placelove)
lightinglove = convert_location(locationlove)
intersectionlove = convert_lighting(lightinglove)
accidenttypelove = convert_intersection(intersectionlove)
seatbeltlove = convert_accident_type(accidenttypelove)
injuredpersonpositionlove = convert_seatbelt(seatbeltlove)
nationalitylove = convert_injured_person_position(injuredpersonpositionlove)
convert_nationality_injured_person(nationalitylove)

