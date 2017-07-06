import os
import re
import json
import pickle
import glob
import datetime
import collections
import pandas as pd
import numpy as np

import sklearn
import sklearn.feature_selection
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt


def error_log_reader(error_file):
    with open(error_file, 'rb') as f:
        data = pickle.load(f)
        
    error_code_df = None
    for key in data:
        if error_code_df is None:
            error_code_df = data[key]
        else:
            error_code_df = error_code_df.append(data[key])

    error_code_df = error_code_df.set_index('processdatetime')
    error_code_df = error_code_df.sort_index()
    return error_code_df

def file2iterator(file):
    print(file)
    with open(file, 'r') as f:
        packages= json.load(f)
    iterator = []
    machine = packages['values'][0]['resourceId']
        
    for pakage in  packages['values'][0]['stat-list']['stat']:
        metric = pakage['statKey']['key']
        times = pakage['timestamps']
        data = pakage['data']
        iterator.extend([(time, machine, metric, value) for time, value in zip(times, data)])
    return sorted(iterator)

def get_start_and_cols(files):
    cols=[]
    starts = []
    for i, file in enumerate(files):

        with open(file, 'r') as f:
            start = np.inf
            packages=json.load(f)
            machine = packages['values'][0]['resourceId']
            l=len(cols)
            for matrix in  packages['values'][0]['stat-list']['stat']:
                start = min(start, min(matrix['timestamps']))
                col = (machine, matrix["statKey"]['key'])
                
                if col not in cols:
                    cols.append(col)
            print(i, machine, len(cols)-l, l, len(cols))
            starts.append((start, file))
    return sorted(starts), cols


def round_to_minutes(time):
    if time.second>=30:
        time = time.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    else:
        time = time.replace(second=0, microsecond=0)
    return time

def round_to_5minutes(time):
    if (time.minute%5)*60 + time.second>=150:
        add_hour = datetime.timedelta(hours=1) if time.minute+(5-(time.minute%5))>=60 else datetime.timedelta(hours=0)
        time = time.replace(minute=(time.minute+(5-(time.minute%5)))%60, second=0, microsecond=0) + add_hour
    else:
        time = time.replace(minute=time.minute-(time.minute%5), second=0, microsecond=0)
    return time



def row_feeder(col_names, iterator):
    row = [None]*len(col_names)
    indeces = {c:i for i, c in enumerate(col_names)}
    
    last_time = None
##    last_yield = None
    

    for time, machine, key, value in iterator:
        time=Unix1000Timestamp2datetime(time)
        time = round_to_5minutes(time)

            
        if (last_time and last_time!=time):
            print(last_time)
            yield last_time, row
##            last_yield = last_time
            row = [None]*len(col_names)
                
        row[indeces[(machine, key)]] = value
        last_time = time

    try:
        yield time, row
    except:
        pass
            

def Unix1000Timestamp2datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time/1000)

def matrix_iterator(file_starts):
    iterators = []
    front_line=[]
    
    start_times = set([t for t, f in file_starts])
    to_launch = min(start_times)
    while (start_times) or (to_launch):
        start_times.remove(to_launch)

        try:
            launched_iter = [iter(file2iterator(f)) for t, f in file_starts if t==to_launch]
            iterators.extend(launched_iter)
            front_line.extend([next(i) for i in launched_iter])
        except StopIteration:
            pass

        to_launch = min(start_times) if start_times else None
        
        while front_line:
            
            to_yield_time = min(front_line)[0]
            
            if to_launch and to_yield_time>=to_launch:
                break

            for index in range(len(front_line)):
                while front_line[index] and front_line[index][0]==to_yield_time:
                    yield front_line[index]

                    try:
                        front_line[index] = next(iterators[index])
                    except StopIteration:
                        iterators[index] = None
                        front_line[index] = None

            iterators = [i for i, f in zip(iterators, front_line) if f]
            front_line = [f for f in front_line if f]

def metric_record_reader(metric_dataframe_file):
    if os.path.isfile(metric_dataframe_file):
        with open(metric_dataframe_file, 'rb') as f:
            metric_record_df = pickle.load(f)
    else:
        machine_matrics = glob.glob(MATRICS_FOLDER)
        file_starts, cols = get_start_and_cols(machine_matrics)
        metric_record_df = pd.DataFrame([[time] + row for time, row in row_feeder(cols, matrix_iterator(file_starts))], columns=['datetime']+cols).set_index('datetime')
        with open(metric_dataframe_file, 'wb') as f:
            pickle.dump(metric_record_df, f)
    return metric_record_df

def event_vectorize(error_code_df, cols):

    col_index = {c:i for i, c in enumerate(cols)}

    row = [0]*len(cols)
    last = None
    for time, event in error_code_df.iterrows():
        time = round_to_5minutes(time)
        event = event['steptypeid']
        if last and last!=time:
            print(last)
            yield last, row
            row = [0]*len(cols)

        row[col_index[event]] +=1
        last=time
    try:
        print(last, sum(row))
        yield last, row
    except:
        pass

def error_vector_df_reader(error_vector_file, error_logs_file):

    if os.path.isfile(error_vector_file):
        with open(error_vector_file, 'rb') as f:
            error_vector_df = pickle.load(f)
    else:         
        error_code_df = error_log_reader(error_logs_file)
        cols = sorted(set(error_code_df['steptypeid']))
        
        error_vector_df = pd.DataFrame([[time] + row for time, row in event_vectorize(error_code_df, cols)], columns=['datetime']+cols).set_index('datetime')

        with open(error_vector_file, 'wb') as f:
            pickle.dump(error_vector_df, f)
    return error_vector_df

def index_period_filtering(df, period):
    df = df[df.index >= min(period)]
    df = df[df.index <= max(period)]
    return df

def plot(array):
    import matplotlib.pyplot as plt
    plt.imshow(array, cmap='seismic')
    plt.colorbar()
    plt.show(block=True)
    plt.clf()

def train_test_split(X, y, train_ratio=0.7):
    splitting_row = int(train_ratio*len(X))
    X_train, X_test =X[:splitting_row], X[splitting_row:]
    y_train, y_test =  y[:splitting_row], y[splitting_row:]

    return X_train, X_test, y_train, y_test

def report(metric_record_df, error_vector_df, col, clf):
    result = pd.concat([metric_record_df, error_vector_df[col]], axis=1, join_axes=[metric_record_df.index])

    X = result.ix[:,:-1]
    y = result.ix[:,-1]

    del result

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.7)
        
    print(sum(y_train==1),sum(y_train==0))
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)

    print('column', col)
    print(classification_report(y_test, y_predict))
    print('\n---\n')

    del X, y, clf, X_train, X_test, y_train, y_test    


def dacay_function(iter_num):
    final_ratio = 0.2
    decay_ratio = 1 - final_ratio
    decay_coeff = 0.2
    return decay_ratio * np.exp(-decay_coeff*iter_num) + final_ratio








if __name__ == '__main__':

    FOLDER =  'D:\\Users\\shanger_lin\\Desktop\\anomilydetection\\'
    MATRICS_FOLDER = FOLDER + 'vrops_data_20170301_20170620\\*\\*.json'
    ERROR_LOG_FILE = FOLDER + 'dbfile'
    METRIC_FILE = FOLDER + 'metrics_df'
    ERROR_VECTOR_FILE = FOLDER + 'error_vactor_df'
    
    error_vector_df = error_vector_df_reader(ERROR_VECTOR_FILE, ERROR_LOG_FILE)
    
    metric_record_df = metric_record_reader(METRIC_FILE)
    metric_record_df = metric_record_df.dropna(axis=1, thresh=int(len(metric_record_df.index)/3))
    metric_record_df = metric_record_df.dropna(axis=0, thresh=int(len(metric_record_df.columns)/3))    
    metric_record_df = metric_record_df.fillna(method='pad').dropna(axis=0, how='any')

    result = pd.concat([metric_record_df, error_vector_df], axis=1, join_axes=[metric_record_df.index])

    




def next_metric_prediction():

    columns = metric_record_df.columns
    index = metric_record_df.index
    VarianceThreshold = sklearn.feature_selection.VarianceThreshold(threshold=1)   
    metric_record_df = VarianceThreshold.fit_transform(metric_record_df)    
    metric_record_df = pd.DataFrame(metric_record_df, columns=[i for i, j in zip(columns, VarianceThreshold.get_support()) if j ], index=index)

    cols = metric_record_df.columns
    LEADNG_TIME_SLOT = 1
    for col in cols:
        X = metric_record_df.drop([col], axis=1)[:-LEADNG_TIME_SLOT]
        y = metric_record_df[col].values.reshape(-1,1)[LEADNG_TIME_SLOT:]

        sigma_threshold = 3
        StandardScaler = sklearn.preprocessing.StandardScaler() 
        y = StandardScaler.fit_transform(y)
        y = (abs(y[:,0])>sigma_threshold)*1

        anomily_number = np.sum(y)
        if anomily_number<100:
##            print(anomily_number, 'bias:', col, )
            continue

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.7)
        if not sum(y_train) or not sum(y_test):
            continue
        
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        target_names = ['others', 'over_{}_sigma'.format(sigma_threshold)]
        report = classification_report(y_test, y_predict, target_names=target_names)

        if float(report.split('\n')[3].split()[1])>0.5:
            print('column', col)
            print(report)
            print('---\n')



def matrics_plot():
    x = metric_record_df.index
    columns = metric_record_df.columns
    for col in columns:
        y = metric_record_df[col]
        file = 'plot{}'.format(os.sep) + re.sub(r'\W+', '',  '_'.join(col)) +'.png' #
        fig = plt.plot(x,y)

        plt.xticks(rotation=45)
        plt.title(col)
        plt.savefig(file)
##        plt.show()
        plt.close()

def error_self_prediction():

    cols = error_vector_df.columns
    error_vector_df = error_vector_df>0
    error_vector_df = error_vector_df * 1

    for col in cols:
        X = error_vector_df.drop([col], axis=1)[:-1]
        y = error_vector_df[col][1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.7)
        
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print('column', col)
        print(classification_report(y_test, y_predict))
        print('---\n')
    
    


def cross_table_experiment():
######################
##    shift=0
##    error_vector_df = pd.DataFrame(np.array(error_vector_df), index=[i+datetime.timedelta(hours=shift) for i in error_vector_df.index], columns=error_vector_df.columns)
######################
    
    start = max([min(metric_record_df.index), min(error_vector_df.index)])
    end = min([max(metric_record_df.index), max(error_vector_df.index)])

    metric_record_df = index_period_filtering(metric_record_df, (start, end))
    error_vector_df = index_period_filtering(error_vector_df, (start, end))


    error_vector_df = error_vector_df>0
    error_vector_df = error_vector_df * 1    
    
    index = metric_record_df.index
    columns = metric_record_df.columns
    
    StandardScaler = sklearn.preprocessing.StandardScaler() 
    metric_record_df = StandardScaler.fit_transform(metric_record_df)
    VarianceThreshold = sklearn.feature_selection.VarianceThreshold(threshold=0)   
    metric_record_df = VarianceThreshold.fit_transform(metric_record_df)    
    metric_record_df = pd.DataFrame(metric_record_df, columns=[i for i, j in zip(columns, VarianceThreshold.get_support()) if j ], index=index)
    
def cov_test():
    cols = error_vector_df.columns
    result = pd.concat([metric_record_df, error_vector_df], axis=1, join_axes=[metric_record_df.index])
    index = result.index
    columns = result.columns
    StandardScaler = sklearn.preprocessing.StandardScaler()
    result = StandardScaler.fit_transform(result)
    result = pd.DataFrame(result, index=index, columns=columns )
    
    a=np.cov(result.T)
    plt.imshow(a, cmap='seismic')
    plt.colorbar()
    plt.show(block=False)
    
    b=a[:,-len(cols):][:-len(cols),:]
    for i, row in enumerate(b.T):
        if max(abs(row))>0.2:
            print(shift, cols[i], sorted(abs(row), reverse=True)[:5])
    print('---')


def combine_table_prediction():
    col = 601
    class_weight  = {1:1/sum(error_vector_df[col]==1), 0:1/sum(error_vector_df[col]==0)}
    clf = RandomForestClassifier()
##    clf = SVC(kernel='rbf', max_iter =100, class_weight=class_weight)
    report(metric_record_df, error_vector_df, col, clf)


        

    



    






















