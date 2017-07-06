import glob
import numpy as np
import pandas as pd


def error_reader(error_file_pattern):

    a=None
    for file in glob.glob(error_file_pattern):
        table = pd.DataFrame.from_csv(file, index_col='ProcessDateTime')
##        print(table['ProcessDateTime'])
##        if 'StepTypeId' in table.columns and 'ProcessDateTime' in table.columns:
##
##            table.set_index('ProcessDateTime', inplace =True)
        if 'StepTypeId' in table.columns:
            if a is None:
                a=table[['StepTypeId', 'Message']]
            else:
                a=a.append(table[['StepTypeId', 'Message']])
            
            time = a.index
##            print(min(time), max(time))
    a= pd.DataFrame(a.values, index = a.index, columns=['StepTypeId', 'Message'])
    a.sort_index(inplace =True)

    return a


if __name__ in '__main__':
    ERROR_FILE_PATTERN = 'D:\\Users\\shanger_lin\\Desktop\\NeuronData\\ACS*'
    error_df = error_reader(ERROR_FILE_PATTERN)

##    failure_fd = error_df[error_df[0]>9000]
##    failure_fd = failure_fd[failure_fd[0]<9999]
