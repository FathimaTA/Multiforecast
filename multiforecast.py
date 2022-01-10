#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import io
import numpy as np
from statistics import mean
import argparse

def reduceResolution(data,res=1):
    index = [res *i for i in range(int(data.shape[0]/res))]
    return data.loc[index,:].reset_index(drop=True)

def forecast(data,dim,model,support_features,target_feature,step):     
    X,Y = [],[]
    for i in range(data.shape[0]-dim-step+1):
        support_x = data.loc[i:i+dim-1,support_features].unstack().ravel()
        target_x = data.loc[i:i+dim-1,target_feature].unstack().ravel()
        X.append(np.concatenate((support_x,target_x)))
        y = data.loc[i+dim-1+step,target_feature].ravel()
        
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y).ravel()
    kf = KFold(n_splits=5)    
    Tr,Ts,Ts_rmse,Ts_mae = [],[], [], []
    for train_index,test_index in kf.split(X,Y):
        model=model.fit(X[train_index],Y[train_index])
        Ts.append(round(model.score(X[test_index],Y[test_index]),2))
        Tr.append(round(model.score(X[train_index],Y[train_index]),2))
        
        pred = model.predict(X[test_index])
        Ts_rmse.append(mean_squared_error(Y[test_index],pred))
        Ts_mae.append(mean_absolute_error(Y[test_index],pred))
    
    return round(mean(Ts),2),round(mean(Ts_rmse),2),round(mean(Ts_mae),2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=30, help='resolution of data in mins')
    parser.add_argument('--dim', type=int, default=4, help='embedding dimension')
    parser.add_argument('--steps', type=int, default=10, help='maximum number of steps ahead for prediction')
    parser.add_argument('--model', type=str, default='MLP', help='regression model MLP/SVR')
    
    args=parser.parse_args() 

    print('data resolution:',args.resolution) 
    print('emb dimension',args.dim)       

    mat = io.loadmat('./PWV_2010from_WS_2_withGradient.mat')
    data = pd.DataFrame(mat.get("DATA", []),columns = ['doy','hour','minute','temperature','solar_radiation','relative_humidity','rain','dew_point_temp','pwv','Unknown1','Unknown2','Unknown3','Unknown4'])
    data['sin_time'] = np.sin(2*np.pi*(data.hour*60+data.minute)/(60.0*24.0))
    data['cos_time'] = np.cos(2*np.pi*(data.hour*60+data.minute)/(60.0*24.0))
    data['sin_doy'] = np.sin(2*np.pi*data.doy/365.0)
    data['cos_doy'] = np.cos(2*np.pi*data.doy/365.0)
    
    data = reduceResolution(data,args.resolution)
    
    
    if args.model == 'MLP': lm = MLPRegressor(max_iter=1000)
    if args.model == 'SVR':lm = SVR()
    
    support_features = ['sin_time','cos_time']
    target_feature = ['solar_radiation']
    
    res = [forecast(data,args.dim,lm,[],target_feature,step) for step in range(1,args.steps)]
    res_time = [forecast(data,args.dim,lm,support_features,target_feature,step) for step in range(1,args.steps)]
    print('results without time')
    print('r2', ' '.join([str(a[0]) for a in res]))
    print('mse', ' '.join([str(a[1]) for a in res]))
    print('mae', ' '.join([str(a[2]) for a in res]))

    print('results with time')
    print('r2', ' '.join([str(a[0]) for a in res_time]))
    print('mse', ' '.join([str(a[1]) for a in res_time]))
    print('mae', ' '.join([str(a[2]) for a in res_time]))
