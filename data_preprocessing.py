import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import xgboost as xgb
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as optim

PATH_ORIGIN = './Input/Original/'
PATH_PROCESSED = './Input/Processed/'
PATH_SUBMISSION = './Input/Submission/'

### Get Haversine distance
city_center = [-8.610989, 41.148932] # Aveiro
def get_dist(lonlat1, lonlat2):
    lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
    lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
    a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]*np.pi/180.0) * np.cos(lonlat2[1]*np.pi/180.0) * np.sin(lon_diff)**2
    d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return(d)

def read_csv_file(file_path, root_path=PATH_ORIGIN):
    file_path = root_path + file_path
    return pd.read_csv(file_path, index_col=0)

def save_csv_file(df, file_path, root_path=PATH_PROCESSED):
    file_path = root_path + file_path
    df.to_csv(file_path, index=True)

def preprocess_data(df):
    # columns: CALL_TYPE,ORIGIN_CALL,ORIGIN_STAND,TAXI_ID,TIMESTAMP,DAY_TYPE,MISSING_DATA, POLYLINE(only in train.csv)
    
    # Replace NULL values with -1 for ORIGIN_CALL and ORIGIN_STAND columns
    df['ORIGIN_CALL'].fillna(-1, inplace=True)
    df['ORIGIN_STAND'].fillna(-1, inplace=True)
    
    # Merge ORIGINAL_STAND and TAXI_ID columns from Input/Original/metaData_taxistandsID_name_GPSlocation.csv
    meta = pd.read_csv(PATH_ORIGIN + 'metaData_taxistandsID_name_GPSlocation.csv')
    df = df.merge(meta, how='left', left_on='ORIGIN_STAND', right_on='ID')
    df.drop(columns=['ID', 'Descricao'], inplace=True)
    
    tqdm.pandas(desc="Calculating distance")
    df['DISTANCE'] = df[['Longitude', 'Latitude']].progress_apply(lambda x: get_dist(x, city_center), axis=1)
    
    # Convert TIMESTAMP to datetime object
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    
    # Extract features from TIMESTAMP column
    df['SECOND'] = df['TIMESTAMP'].dt.second
    df['MINUTE'] = df['TIMESTAMP'].dt.minute
    df['HOUR'] = df['TIMESTAMP'].dt.hour
    df['DAY'] = df['TIMESTAMP'].dt.day
    df['MONTH'] = df['TIMESTAMP'].dt.month
    df['DAYOFWEEK'] = df['TIMESTAMP'].dt.dayofweek # 0 is Monday, 6 is Sunday
    df.drop(columns=['TIMESTAMP'], inplace=True)

    # One-hot encoding for CALL_TYPE and DAYTYPE columns
    df = pd.get_dummies(df, columns=['CALL_TYPE', 'DAY_TYPE'])
    
    # Process POLYLINE column and calculate travel time if POLYLINE exists
    if 'POLYLINE' in df.columns:
        # tqdm.pandas(desc="Processing POLYLINE")
        
        # transform polyline string to list of points
        tqdm.pandas(desc="Processing POLYLINE")
        df['POLYLINE'] = df['POLYLINE'].progress_apply(lambda x: json.loads(x))    
        
        tqdm.pandas(desc="Calculating count")
        #count how many points are in each polyline
        df['POLYLINE_COUNT'] = df['POLYLINE'].progress_apply(lambda x: len(x))
        
        tqdm.pandas(desc="Calculating travel time")
        df['TRAVEL_TIME'] = df['POLYLINE_COUNT'].progress_apply(lambda x: (x - 1) * 15)
        
        df.drop(columns=['POLYLINE_COUNT', 'POLYLINE'], inplace=True)

    #set type to float32
    df = df.astype('float32')
    
    return df

def split_data(df, test_size=0.2, random_state=151):
    train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, val

def extract_features_and_labels(df, label_column):
    # Extract features and labels
    if label_column:
        X = df.drop(columns=[label_column])
        y = df[label_column]
        return X, y
    else:
        return df
    
def get_all_data():
    # try to read processed data from csv files
    try:
        df_train = read_csv_file('train.csv', PATH_PROCESSED)
        df_test = read_csv_file('test.csv', PATH_PROCESSED)
        print('Processed data found. Reading processed data...')
    except:
        print('Processed data not found. Reading original data...')
        # Read data from csv files
        df_train = read_csv_file('train.csv')
        df_test = read_csv_file('test_public.csv')
    
        # Preprocess data
        df_train = preprocess_data(df_train)
        df_test = preprocess_data(df_test)
        
        # Save processed data to csv files
        df_train.to_csv(PATH_PROCESSED + 'train.csv')
        df_test.to_csv(PATH_PROCESSED + 'test.csv')
    
    # Split data into train and validation sets
    train, val = split_data(df_train)
    
    # Extract features and labels
    X_train, y_train = extract_features_and_labels(train, 'TRAVEL_TIME')
    X_val, y_val = extract_features_and_labels(val, 'TRAVEL_TIME')
    X_test = extract_features_and_labels(df_test, None)
    
    return X_train, y_train, X_val, y_val, X_test

def process_train_or_val(X, y, drop_nan=True):
    # drop rows with y > 10000 in both X and y
    X = X[y < 10000]
    y = y[y < 10000]
    
    if drop_nan:
        # drop rows with NaN values in X
        X = X.dropna()
        y = y[X.index]
    else:
        # fill NaN values in X with mean or maybe median
        X = X.fillna(X.mean())
    
    return X, y
    
def generate_submission_file(model, X_test, file_path=None, root_path=PATH_SUBMISSION, save=False):
    
    df = read_csv_file('sampleSubmission.csv')
    
    try:
        y_pred = model.predict(X_test) # for sklearn models
    except:
        y_pred = model(X_test) # for pytorch Neural Network
        # it is tensor, so we need to convert it to numpy array
        # use Tensor.cpu() to copy tensor to host memory first
        y_pred = y_pred.cpu().detach().numpy()
    df['TRAVEL_TIME'] = y_pred
    if save and file_path != None:
        save_csv_file(df, file_path, root_path)
        
    return df

def convert_df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(df.values, dtype=torch.float32)

def convert_all_df_to_tensor(X_train, X_val, X_test, y_train, y_val):
    cuda = True if torch.cuda.is_available() else False
    try:
        if cuda:
            # convert the data to tensor
            train_data = torch.tensor(X_train.values, dtype=torch.float32).cuda()
            val_data = torch.tensor(X_val.values, dtype=torch.float32).cuda()
            test_data = torch.tensor(X_test.values, dtype=torch.float32).cuda()

            # convert the target to tensor
            train_target = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).cuda()
            val_target = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).cuda()
        else:
            # convert the data to tensor
            train_data = torch.tensor(X_train.values, dtype=torch.float32)
            val_data = torch.tensor(X_val.values, dtype=torch.float32)
            test_data = torch.tensor(X_test.values, dtype=torch.float32)
            
            # convert the target to tensor
            train_target = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            val_target = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    except:
        if cuda:
            # convert the data to tensor
            train_data = torch.tensor(X_train, dtype=torch.float32).cuda()
            val_data = torch.tensor(X_val, dtype=torch.float32).cuda()
            test_data = torch.tensor(X_test, dtype=torch.float32).cuda()

            # convert the target to tensor
            train_target = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).cuda()
            val_target = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).cuda()
        else:
            # convert the data to tensor
            train_data = torch.tensor(X_train, dtype=torch.float32)
            val_data = torch.tensor(X_val, dtype=torch.float32)
            test_data = torch.tensor(X_test, dtype=torch.float32)
            
            # convert the target to tensor
            train_target = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            val_target = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
    
    return train_data, val_data, test_data, train_target, val_target