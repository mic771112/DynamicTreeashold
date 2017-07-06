#!/usr/bin/python3
#
#
#
#
#
#
#
#SgangerLin 2017
#
import os
import pickle
import glob
import json
import collections
import datetime
import pandas as pd
import numpy as np
import logging
import sklearn
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.decomposition
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class dynamic:
    
    def __init__(self):
        pass

    gaussian = lambda self, x, mu, sigma:  np.exp( -((x-mu)**2)/(2*(sigma**2)) ) / (np.sqrt(2*np.pi*(sigma**2)))
    
    def _preprocess(self, df, mode = 'transfrom'):

        indeces = df.index
        
        if mode == 'transform':
            df =self.preprocessor.transform(df) 
        elif mode == 'fit_transform':
            self.preprocessor =  Pipeline(
                steps=[
                    ('StandardScaler', sklearn.preprocessing.StandardScaler()),
                    ('VarianceThreshold', sklearn.feature_selection.VarianceThreshold(threshold=1)),
                    ('PCA', sklearn.decomposition.PCA())
                    ]
                )
            df =self.preprocessor.fit_transform(df)
        else:
            raise TypeError('unknow mode set {}'.format(mode))

        df = pd.DataFrame(df, index=indeces)
        
        return df

    def fit(self, df):
        
        df = self._preprocess(df, mode = 'fit_transform')
        time_distributed_data = collections.defaultdict(list)
        for k, v in df.iterrows():
            time_distributed_data[k.hour].append(v)
        for time in time_distributed_data:
            time_distributed_data[time]=np.array(time_distributed_data[time])

        self.means = {k:np.mean(time_distributed_data[k], axis=0) for k in time_distributed_data}
        self.stds = {k:np.std(time_distributed_data[k], axis=0) for k in time_distributed_data}

        return df


    
    def predict(self, to_predict, diamentions = 2, anomly_ratio=0.1):
        
        to_predict = self._preprocess(to_predict, mode = 'transform')
        possibility = np.array([ np.prod(self.gaussian(v[:diamentions], self.means[k.hour][:diamentions], self.stds[k.hour][:diamentions])) for k, v in to_predict.iterrows()])  
        predict = np.array([0 if i >= sorted(possibility)[int(len(possibility)*anomly_ratio)] else 1 for i in possibility])

        return predict
    


if __name__ == '__main__':
    
    FILES = glob.glob('D:\\Users\\shanger_lin\\Desktop\\NeuronData\\ipp-poc-vrops-1-2017-06-07\\*\\*')   
    DATAFRAME = 'D:\\Users\\shanger_lin\\Desktop\\NeuronData\\ipp-poc-vrops-1-2017-06-07\\dynamictable'    

    logging.basicConfig(level=logging.INFO)
    logging.info('start')

    df = data_frame_reader(FILES, DATAFRAME)
    df = df.fillna(method='pad').dropna(axis=0, how='any')

    model = dynamic()
    transformed_df = model.fit(df)
    predict = model.predict(df)

    indeces = transformed_df.index
    color = np.array(['r' if i else 'b' for i in predict])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    array = np.array(transformed_df)
    ax.scatter([i.hour + i.minute/60 for i in indeces], array[:,0], array[:,1], s = 1, c=color)
    plt.show(block=False)

    
    
    
