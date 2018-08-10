# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:10:48 2018

@author: Wahyu Nainggolan
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from fancyimpute import SoftImpute,KNN 
import pandas as pd

def encode_dataset_by_onehot_encoder(labels):
    enc = OneHotEncoder(sparse=False)
    integer_encoded = np.array(labels).reshape(-1)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = enc.fit_transform(integer_encoded)

    return onehot_encoded

Hasil=[]
def preprocessingData(dataset):
    # solve missing value
    dataset.iloc[:,5:] = SoftImpute().complete(dataset.iloc[:,5:])
    dataset_independent = dataset.round({'HTRF':0,'THFP':0, 'PHPAR':0, 'PHSPAR':0, 'PHPSAR':0, 'PHDAR':0, 'PHTSA':0, 'PHDBAR':0,
       'PHAAR':0, 'ATRF':0, 'TAFP':0, 'PAPAR':0, 'PASPAR':0, 'PAPSAR':0, 'PADAR':0, 'PATSA':0,
       'PADBAR':0, 'PAAAR':0})
    dataset_independent=dataset_independent.drop('Hasil',axis=1)    
    #label encoder
    dataset_dependent = dataset.iloc[:,[4]].values
    labelencoder_X=LabelEncoder()
    dataset_dependent=labelencoder_X.fit_transform(dataset_dependent)
    dataset_dependent_baru=pd.DataFrame(dataset_dependent,columns=['Hasil'])
    return dataset_independent,dataset_dependent_baru

def preprocessingData_pialadunia2018(dataset):
    # solve missing value
    dataset.iloc[:,2:] = SoftImpute().complete(dataset.iloc[:,2:])
    dataset_independent = dataset.round({'HTRF':0,'THFP':0, 'PHPAR':0, 'PHSPAR':0, 'PHPSAR':0, 'PHDAR':0, 'PHTSA':0, 'PHDBAR':0,
       'PHAAR':0, 'ATRF':0, 'TAFP':0, 'PAPAR':0, 'PASPAR':0, 'PAPSAR':0, 'PADAR':0, 'PATSA':0,
       'PADBAR':0, 'PAAAR':0})
    dataset_independent=dataset_independent.drop('Hasil',axis=1)    
    #label encoder
    dataset_dependent=dataset['Hasil']
    return dataset_independent,dataset_dependent


def preprocessingData_toDBN(dataset):
    # solve missing value
    dataset.iloc[:,5:] = KNN(3).complete(dataset.iloc[:,5:])
    dataset_independent = dataset.round({'HTRF':0,'THFP':0, 'PHPAR':0, 'PHSPAR':0, 'PHPSAR':0, 'PHDAR':0, 'PHTSA':0, 'PHDBAR':0,
       'PHAAR':0, 'ATRF':0, 'TAFP':0, 'PAPAR':0, 'PASPAR':0, 'PAPSAR':0, 'PADAR':0, 'PATSA':0,
       'PADBAR':0, 'PAAAR':0})
    dataset_independent=dataset_independent.drop('Hasil',axis=1)    
    #label encoder
    dataset_dependent = dataset.iloc[:,[4]].values
    labelencoder_X=LabelEncoder()
    dataset_dependent=labelencoder_X.fit_transform(dataset_dependent)
    dataset_dependent_baru=pd.DataFrame(dataset_dependent,columns=['Hasil'])
    return dataset_independent,dataset_dependent_baru

# one hot encoder 
def preprocessingData_onehotencoder(dataset_dependent):
    dataset_dependent=encode_dataset_by_onehot_encoder(dataset_dependent)
    return dataset_dependent

# normalisasi
def normalisasi(data):
    data_max=max(data)
    data_min=min(data)
    normalized = (data - data_min) / (data_max - data_min)
    return normalized

