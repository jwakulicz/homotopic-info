#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import vmmp_gmm.pickle_helper as pickle_helper

# %%
def column_function(dataframe=None, columns=None, function=None, **kwargs):
    """
    Computes an algebraic function of specified columns of a dataframe. 

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to be transformed.
    columns : list<object>
        A list of column labels, corresponding to varables of the function.
        The list may be any subset of the columns in the dataframe (assuming they are) numerical
        columns suitable for algebraic operations. The list order corresponds to order of the 
        var_labels supplied to the function. All column names must exist in dataframe.  
    function : function
        The algebrraic function to be evaluated. This may be a custom function, or e.g. a standard
        numpy function. For example, let:

                df =     A  B   C
                    0   1   4   7
                    1   2   5   8
                    2   3   6   9 

        then we may call

                column_function(df, columns=['A', 'B', 'C'], function=np.median)

        which returns
                        A   B   C   output
                    0   1   4   7   4.0
                    1   2   5   8   5.0
                    2   3   6   9   6.0

    **kwargs : arbitrary keword arguments. This permits arbitrary keyword arguments to be passed to
        this function. These will then be passed to the function. The function will consume these
        in the form of a kwarg dictionary. For example, a custom function with arbitrary parameters may be
        constructed as follows:

                def linear_combination(v,**kwargs):  
                    return  np.dot(v, kwargs['weights'])

        then we may call

                column_function(df, columns = ['A','B','C'], function = linear_combination, weights = [1,2,3])
        
        which returns

                        A   B   C   output
                    0   1   4   7   30
                    1   2   5   8   36
                    2   3   6   9   42

    """

    assert type(dataframe) == pd.DataFrame, 'dataframe must be a pd.DataFrame object.'
    assert all([column in dataframe.columns for column in columns]), 'All column names must be valid dataframe column names.'
    assert callable(function), 'function must be a callable function.'
    df = dataframe.copy()
    assert df.empty==False, 'dataframe is empty, so cannot compute function.'

    df['output'] = df.apply(lambda x: function([x[c] for c in columns], **kwargs), axis=1)
    return df

def get_vec(v): 
    return np.array([v[0], v[1]])

#%%
header_list = ['time_stamp', 'pid', 'pos_x', 'pos_y', 'pos_z', 'vel', 'motion_angle', 'heading']
raw_df = pd.read_csv('./atc_mall/atc-20121028.csv', names=header_list)

#%%
pids = raw_df.pid.unique()
trace_dict = {}
# df = pd.DataFrame(columns=['time_stamp'])

for n, pid in tqdm(enumerate(pids)):
    that_guy_df = raw_df[raw_df['pid'] == pid]
    that_guy_df = column_function(dataframe=that_guy_df, columns=['pos_x','pos_y'], function=get_vec)
    that_guy_df = that_guy_df.drop(columns = ['pos_x','pos_y', 'pos_z', 'vel', 'motion_angle', 'heading'])
    that_guy_df = that_guy_df.pivot(index='time_stamp', columns='pid', values='output').reset_index()
    trace_dict[str(pid)] = that_guy_df
    # columns are time stamps, rows are pids, values are 2x1 np.arrays 
    
    # df = pd.merge(df, that_guy_df, on='time_stamp', how='outer') takes too long

#%%
pickle_helper.pickle_save_data(data = trace_dict, folder_path='./atc_mall/', file_name='processed_data_20121028')


# %%
test = pickle_helper.pickle_load_data(folder_path='./atc_mall', file_name='processed_data_20121028')
# %%