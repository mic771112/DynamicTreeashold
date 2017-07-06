import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Threshold:

    def __init__(self, columns, init_array, timeresolution=datetime.timedelta(minutes=5), period=datetime.timedelta(weeks=1), init_weight = 5):
        import datetime
        import numpy as np
        import pandas as pd
        t0 = datetime.datetime(1,1,1)
        time_tuple_extracter = lambda t: (t.isoweekday(), datetime.time(t.hour, t.minute))
        indeces = np.array( sorted([time_tuple_extracter(t0+timeresolution*step) for step in range(int(period/timeresolution)) ]))

        self.time_resolution = timeresolution
        self.indeces = indeces
        self.columns = columns

        self.indeces_indexer = { tuple(j) :i for i, j in enumerate(indeces)}
        self.indeces_columns = { tuple(j) :i for i, j in enumerate(columns)}

        m =  len(indeces) 
        n = len(columns)

        self.timetuplecount = m
        self.columnscount = n

        self.time_tuple_extracter = time_tuple_extracter
        
        self.count = np.zeros((m,n))
        
        self.mean = np.zeros((m,n))
        self.mean[:,:]  = np.nanmean(init_array,axis=0)
        
        self.upper =  np.zeros((m,n))
        self.upper[:,:]  = np.nanvar(init_array,axis=0)
        
        self.lower = np.zeros((m,n))
        self.lower[:,:]  = np.nanvar(init_array,axis=0)
        
        self.upper_weight = np.ones(n) * init_weight
        self.lower_weight = np.ones(n) * init_weight  

    def update(self, times, rows, learning_rate=0.05, gradient_dacay_rate = 1, sn_limit=5, updating_range=range(-1,2)):

        if len(rows.shape)==1:
            rows = (rows,)
            times = (times,)

        times = np.vectorize(self.round_times)(times, self.time_resolution)

        for time, row in zip(times, rows):
            current_time_tuple = self.time_tuple_extracter(time)
            current_index = self.indeces_indexer[current_time_tuple]
            target_indeces = [(current_index+i)%self.timetuplecount for i in updating_range]

            # NaN value preprecess
            row[np.isnan(row)] = self.mean[current_index,:][np.isnan(row)]

            # activation_check
            sn_ratio = 2 * (row-self.mean[current_index,:])**2 / (self.upper[current_index,:] + self.lower[current_index,:])
            over_anomaly = sn_ratio < sn_limit
            activate = np.isfinite(row) * over_anomaly  #* (self.count[current_index,:]<1)
            
            # get informations
            square_difference = (row-self.mean[target_indeces,:])**2
            updating_rate = learning_rate #((1-learning_rate) * np.exp(-self.count[current_index,:] * gradient_dacay_rate) + learning_rate)
            
            #updating
            self.mean[target_indeces,:] +=  activate * updating_rate * (row - self.mean[target_indeces,:])

            activate_uppers = ((row > self.mean[target_indeces,:])*activate) *1
            self.upper[target_indeces,:] +=  activate_uppers*(updating_rate*(square_difference - self.upper[target_indeces,:]))

            activate_lowers = ((row < self.mean[target_indeces,:])*activate) *1
            self.lower[target_indeces,:] +=  activate_lowers*(updating_rate*(square_difference - self.lower[target_indeces,:]))

            self.count[current_index,:] += activate

    def detect(self, times, rows):

        if len(rows.shape)==1:
            rows = np.array((rows,))
            times = np.array((times,))
        times = np.vectorize(self.round_times)(times, self.time_resolution)

        current_time_tuples = [self.time_tuple_extracter(t) for t in times]
        current_indeces = [self.indeces_indexer[t] for t in current_time_tuples]

        mean = self.mean[current_indeces,:]
        
        upper_bound = mean + self.upper_weight * np.sqrt(self.upper[current_indeces,:])
        lower_bound = mean - self.lower_weight * np.sqrt(self.lower[current_indeces,:])
        
        upper_break = rows > upper_bound
        lower_break = rows < lower_bound

        breaking = np.logical_or(upper_break, lower_break)

        return times, breaking


    def update_detect(self, times, rows, learning_rate=0.05, gradient_dacay_rate = 0.05, sn_limit=10, updating_range=range(-1,2)):
        self.update(times, rows, learning_rate, gradient_dacay_rate, sn_limit, updating_range)
        times, breaking = self.detect(times, rows)
        return times, breaking
    
    def round_times(self, time, resolution=datetime.timedelta(minutes=5)):
        resolution_seconds = resolution.total_seconds()
        remainder_seconds = time.timestamp() % resolution_seconds
        if remainder_seconds >= resolution_seconds/2:
            time += datetime.timedelta(0,resolution_seconds - remainder_seconds)
        else:
            time -= datetime.timedelta(0,remainder_seconds)

        return time






if __name__ == '__main__':
    import ipp_readers
    FOLDER =  'D:\\Users\\shanger_lin\\Desktop\\anomilydetection\\'
    MATRICS_FOLDER = FOLDER + 'vrops_data_20170301_20170620\\*\\*.json'
    ERROR_LOG_FILE = FOLDER + 'dbfile'
    METRIC_FILE = FOLDER + 'metrics_df'
    ERROR_VECTOR_FILE = FOLDER + 'error_vactor_df'
##    error_vector_df = error_vector_df_reader(ERROR_VECTOR_FILE, ERROR_LOG_FILE)
    
    metric_record_df = ipp_readers.metric_record_reader(METRIC_FILE)
    metric_record_df = metric_record_df.dropna(axis=1, thresh=int(len(metric_record_df.index)/3))
    metric_record_df = metric_record_df.dropna(axis=0, thresh=int(len(metric_record_df.columns)/3))    
##    metric_record_df = metric_record_df.fillna(method='pad').dropna(axis=0, how='any')

    time_tuple_extracter = lambda t: (t.isoweekday(), datetime.time(t.hour, t.minute))
    
    columns = metric_record_df.columns

    plotting_step = 1
    init_rows = 2016
    init_array = metric_record_df.ix[:init_rows,:].values
    T = Threshold(columns=columns, init_array=init_array)
    
    indeces = pd.MultiIndex.from_tuples([i for i in T.indeces], names=['weekday', 'time'])
    indeces_indexer = {tuple(tup):i for i, tup in enumerate(T.indeces)}
    
    plt.ion()
    
    anomily = np.zeros([len(indeces),len(columns)]) #pd.DataFrame(, columns=columns, index=indeces)
    data = np.zeros([len(indeces),len(columns)])

    for step, (t, r) in enumerate(metric_record_df.ix[init_rows:,:].iterrows()):

        target_column = int(step/plotting_step)%len(columns)
        
        rows = r.values
        
        times, breaking = T.update_detect(t, rows)
        
        time_tuples = [time_tuple_extracter(t) for t in times]
        time_tuple_indeces = [indeces_indexer[t] for t in time_tuples]
        anomily[time_tuple_indeces,:] = breaking
        data[time_tuple_indeces,:] = rows
        
        uw = T.upper_weight[target_column]
        lw = T.lower_weight[target_column]

##        T.update(t, rows)


        
        if (step > len(indeces)) and (not step%plotting_step):

            y = T.mean[:,target_column]
            up = T.upper[:,target_column]
            down = T.lower[:,target_column]
            col_breaking = anomily[:,target_column]
            col_data = data[:,target_column]
            plt.clf()



##            mean = pd.Series(data=y, index=indeces)
##            upper = pd.Series(data=y+np.sqrt(up), index=indeces)
##            lower = pd.Series(data=y-np.sqrt(down), index=indeces)
####            breaking_series = pd.Series(data=col_breaking, index=indeces)
####            data_series = pd.Series(data=col_data, index=indeces)
##
##            df = pd.DataFrame(np.array([col_data,col_breaking, indeces]).T, columns=['a','b','c'], index=indeces)
##
##             x='a', y='b', c='c'
##            mean.plot(rot=15)
##            upper.plot(rot=15)
##            lower.plot(rot=15)
##            df.plot(df, rot=15, kind='scatter', x='c', y='a', c='b')
            ind = range(len(T.indeces))
            plt.title(columns[target_column])
            plt.plot(ind, y, 'g-' , linewidth=0.3)
            plt.plot(ind, y+uw*np.sqrt(up), '-.', linewidth=0.3)
            plt.plot(ind, y-lw*np.sqrt(down), ':', linewidth=0.3)
            plt.scatter(ind, col_data, c=['r' if i else 'b' for i in col_breaking], s=1)            
            plt.draw()
            plt.savefig('working_plot{}.png'.format(os.sep + re.sub(r'\W+', '',  '_'.join(columns[target_column]))))
            plt.pause(0.00001)
            

    print('finished')







