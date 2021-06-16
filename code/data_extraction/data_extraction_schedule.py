from data_extraction.data_extraction_cpt import data_extraction_cpt
from data_extraction.data_extraction_sitec import data_extraction_sitec
from multiprocessing import Pool
import dask
dataset_cpt = 'CPT'
dataset_sitec = 'SITEC'

def data_extraction_schedule(v_dask_data_extraction, raw_files, pool_size, dataset):
      if dataset == dataset_cpt:
          data_extraction = data_extraction_cpt
      if dataset == dataset_sitec:
          data_extraction = data_extraction_sitec
      print('1. Data extraction')
      if not v_dask_data_extraction:
          with Pool(pool_size) as p:
              p.map(data_extraction, reversed(raw_files))
      else:
          delayed_results = []
          for file_one in reversed(raw_files):
              delayed_results.append(dask.delayed(data_extraction)(file_one))
          dask.compute(delayed_results)#, scheduler='processes')
