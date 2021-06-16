import pandas as pd
from functools import wraps
from time import time
import datetime
import gc
import psutil
import os 
import sys
#from memory_profiler import profile
import dask.dataframe as dd
import glob
import dask
import hashlib
from config import debug_mode, write_monitoring, store_results
from tools.signature import GLOBAL_SIGNATURE

def mem():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss
    gb =round(memory_usage/ (1024.0 ** 3),2)
    return gb

#@profile
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        time_start = datetime.datetime.fromtimestamp(ts)
        memory_usage_start = mem()
        print(time_start.strftime('%Y-%m-%d %H:%M:%S'), '[START]', f.__name__)
        skip_store_result_list = ['data_extraction_cpt', 'gen_error_code_indicators', 'remove_error_codes', 'calculate_statistics', #'extract_windows_and_features',
                                  'extract_examples', 'remove_global_timestamp',#'standardize_features',
                                  'get_x_y', 'eval', 'eval_ensemble']
        if store_results and f.__name__ not in skip_store_result_list:
            result = create_or_load_result(f, *args, **kw) 
        else:
            result = f(*args, **kw)
        te = time()
        time_end = datetime.datetime.fromtimestamp(te)
        #gc.collect()
        memory_usage_end = mem()
        memory_disp = str(memory_usage_start) + ' GB > ' + str(memory_usage_end) + ' GB; df size: ' 
        df_mem = str(round(sys.getsizeof(result)/1024**3, 2)) 
        if isinstance(result, pd.DataFrame) or isinstance(result, dask.dataframe.core.DataFrame):
            df_disp = "columns: " + str(len(result.columns))
        else:
            df_disp = ''
        print (time_end.strftime('%Y-%m-%d %H:%M:%S'), '[END] %r  %2.0f min' % (f.__name__,  round((te-ts)/60)), memory_disp,  df_mem + ' GB;', df_disp )
        if write_monitoring:
            df = pd.DataFrame({ 'ts': [ts], 'te': [te], 'ts-te':[te-ts], 'function_name':[f.__name__],'memory_usage_start': memory_usage_start, 'memory_usage_end':  memory_usage_end, 'data_memory': df_mem })
            df.to_csv('../monitoring/time.csv', mode='a', index = False, header=None, sep=";", decimal=",")
            if debug_mode and ((isinstance(result, pd.DataFrame) or isinstance(result, dask.dataframe.core.DataFrame)) ):
                debug_name = '../monitoring/debug/'+f.__name__ + '.csv'
                result.compute().to_csv(debug_name, sep=";", decimal=",")#, engine='xlsxwriter') # pip install xlsxwriter
                debug_name_memory = '../monitoring/memory/' + f.__name__ + '.csv'
                result.compute().memory_usage(deep=True).to_csv(debug_name_memory, sep=";", decimal=",")
        return result
    return wrap

def create_or_load_result(f, *args, **kw):
    kwargs_repr = [f"{k}_{v}" for k, v in kw.items()]
    kwargs_repr = "_".join(kwargs_repr)
    if f.__name__ != 'create_global_dataframe':
        signature_new = GLOBAL_SIGNATURE.sig + f.__name__ + kwargs_repr
    else:
        signature_new = GLOBAL_SIGNATURE.sig + f.__name__ + kwargs_repr + str(sorted(glob.glob(args[0])))
    GLOBAL_SIGNATURE.sig = f.__name__ + '_' + hashlib.md5(signature_new.encode('utf-8')).hexdigest()
    print('Signature after {}: {}'.format(f.__name__, GLOBAL_SIGNATURE.sig))
    file_path = '../data/Intermediate_Results\\' + GLOBAL_SIGNATURE.sig + str('.parquet.gzip')
    available_paths = glob.glob('../data/Intermediate_Results/*.gzip')
    if file_path in available_paths:
        if f.__name__ not in ['extract_windows_and_features', 'standardize_features']:
            result = dd.read_parquet(file_path)
            print('parquet read')
        if f.__name__ in ['extract_windows_and_features', 'standardize_features']:
            result = pd.read_parquet(file_path)
            print('pandas read')
    else:
        result = f(*args, **kw)
        if f.__name__ not in ['extract_windows_and_features', 'standardize_features']:
            result.to_parquet(file_path)
            print('parquet created')
        if f.__name__ in ['extract_windows_and_features', 'standardize_features']:
            result.to_parquet(file_path)
            print('pandas created')
    return result   