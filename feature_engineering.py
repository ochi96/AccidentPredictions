import os
import numpy as np
import string
from matplotlib import pyplot as plt 
from matplotlib import style
import pandas as pd 
import pandas_profiling as pp
import matplotlib.pyplot as partial_train_data
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder,StandardScaler, MinMaxScaler

def view_data():
    raw_data = pd.read_csv('Data/accident_data.csv', delimiter=',', engine='python')
    print(raw_data.columns.tolist())
    '''profile = pp.ProfileReport(raw_data, title='Pandas Profiling Report', explorative=True)
    profile.to_file("my_report.html")'''
    #From the generated report, the following columns were dropped due to no impact on the severity of injuries.
    #they are mostly uniform data.
    raw_data = raw_data.drop(['Emirate','City','Road','Report type','Computed','Computed2','Streets',
                'Gender of the injured','Fasten seat belt','Year','Month', 'Day','Report Number',
                'Inp Age','Date - Month', 'Age Group - Ministry', 'Time', 'Day.1',
                'Age Group', 'Weather','Road surface', 'Area','Block',
                'Location.1', 'Nationalities'], axis=1)       
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]
    print(raw_data.columns.tolist())
    return raw_data


def convert_gender(data):
    raw_data = data
    genders = pd.get_dummies(raw_data['Gender'], prefix='Gender')
    raw_data = pd.concat([raw_data,genders],axis=1) 
    raw_data = raw_data.drop('Gender', axis=1)

    return  raw_data

def convert_week(data):                    
    raw_data = data
    weeks = pd.get_dummies(raw_data['Week'], prefix='Week')
    raw_data = pd.concat([raw_data,weeks], axis=1) 
    raw_data = raw_data.drop('Week', axis=1)
    #to get only one date-time per entry
    raw_data['Date_time'] = raw_data['Date'].astype(str)+' '+raw_data['Iac Rep Time']
    #drop the separate date and time
    raw_data = raw_data.drop(['Date','Iac Rep Time'],axis=1)
    #change the format to datetime
    raw_data['Date_time'] = pd.to_datetime(raw_data['Date_time'])
    raw_data['Date_time'] = raw_data['Date_time'].dt.strftime('%d%H%M')
    raw_data['Date_time'].str.replace('[{}]'.format(string.punctuation), '')
    raw_data['Date_time'] = raw_data['Date_time'].astype(int)

    print(raw_data["Date_time"].head(10))

    return raw_data

def convert_reasons(data):
    raw_data = data
    reasons = pd.get_dummies(raw_data['Reasons'], prefix='Reasons_')
    raw_data = pd.concat([raw_data,reasons],axis=1)
    raw_data = raw_data.drop(['Reasons'], axis=1)
    return raw_data

def convert_stations(data):
    raw_data = data
    stations = pd.get_dummies(raw_data['Police Station'], prefix='_')
    raw_data = pd.concat([raw_data,stations],axis=1)
    raw_data = raw_data.drop('Police Station', axis=1)
    return raw_data

def convert_street(data):
    raw_data = data
    raw_data['Street'] = raw_data['Street'].astype(str)
    #too many streets, convert to only street and intersection
    Street_Dictionary = {'intersection':'Intersection','Intersection':'Intersection', 'Street':'Street','street':'Street' }
    raw_data['Street']= raw_data['Street'].str.extract('(intersection)',expand=False)
    raw_data['Street_name'] = raw_data['Street'].map(Street_Dictionary).fillna('Street')
    #print(raw_data['Street_name'].head(20))
    Street_dummies = pd.get_dummies(raw_data['Street_name'],prefix='Streetname')
    raw_data = pd.concat([raw_data,Street_dummies], axis=1)
    raw_data = raw_data.drop(['Street','Street_name'], axis=1)
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

    types = pd.get_dummies(raw_data['lighting'], prefix='lighting_')
    raw_data = pd.concat([raw_data, types], axis=1)
    raw_data = raw_data.drop(['lighting'],axis=1)
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
    raw_data = pd.concat([raw_data,types], axis=1) 
    raw_data = raw_data.drop('Accident Type', axis=1)

    return raw_data

def convert_seatbelt(data):
    raw_data = data
    types = pd.get_dummies(raw_data["Seat Belt"], prefix='seat belt')
    raw_data = pd.concat([raw_data,types],axis=1)
    raw_data = raw_data.drop(['Seat Belt'], axis=1)

    return raw_data

def convert_injured_person_position(data):
    raw_data = data
    types = pd.get_dummies(raw_data["Injured person's seat"], prefix='seat')
    raw_data = pd.concat([raw_data,types],axis=1) 
    types2 = pd.get_dummies(raw_data['Injured person position'], prefix='position')
    raw_data = pd.concat([raw_data,types2],axis=1) 
    raw_data = raw_data.drop(['Injured person position',"Injured person's seat" ], axis=1)

    return raw_data

def convert_nationality_injured_person(data):
    raw_data = data
    nationalities = pd.get_dummies(raw_data['Nationality of the Injured person'], prefix='nationality')
    raw_data = pd.concat([raw_data, nationalities],axis=1) 
    raw_data = raw_data.drop(['Nationality of the Injured person'], axis=1)

    actions = {'None':0, 'Cross but not on the crosswalk ':1,'Cross on crosswalk not in the intersection':1,
                'Cross on crosswalkS not in the intersection':2,"Cross, there's no crosswalk":1, 'Stop':0, 'Cross without attention':1,
                'Coss, but not on the crosswalk':1, 'Coss on the crosswalk of the intersection':1,
                'Cross on crosswalk of the intersection':1, 'Cross on crosswalk, not in the intersection':1,
                'Cross on the crosswalk of the intersection':0, 'Stop on the road strip':1}
    raw_data['Pedestrian action'] = raw_data['Pedestrian action'].map(actions).fillna(1)
    
    raw_data['Number of Lanes']=raw_data['Number of Lanes'].fillna(3.36)

    return raw_data

def normalize(data):
    raw_data = data
    train_targets = raw_data['Degree of the injury']
    embedings_data = raw_data['Accident Description']
    train_data = raw_data.drop(['Degree of the injury'],axis=1)
    sc = StandardScaler()
    #mmsc = MinMaxScaler()
    dataset_numerical_features = list(train_data.select_dtypes(include=['int64','float64','int32','float32']).columns)
    dataset_scaled = pd.DataFrame(data=train_data)
    dataset_scaled[dataset_numerical_features] = sc.fit_transform(dataset_scaled[dataset_numerical_features])
    print(train_data.shape)
    train_data.to_csv('train_data9.csv', index=False)
    train_targets.to_csv('train_targets9.csv', index = False)
    return train_data, train_targets

def final_features():
    clean = view_data()
    week = convert_gender(clean)
    reason = convert_week(week)
    station = convert_reasons(reason)
    street = convert_stations(station)
    place = convert_street(street)
    location = convert_place(place)
    lighting = convert_location(location)
    intersection = convert_lighting(lighting)
    accidenttype = convert_intersection(intersection)
    seatbelt = convert_accident_type(accidenttype)
    injuredpersonposition = convert_seatbelt(seatbelt)
    nationality = convert_injured_person_position(injuredpersonposition)
    normalizedata = convert_nationality_injured_person(nationality)
    final = normalize(normalizedata)

    return final

final_features()






