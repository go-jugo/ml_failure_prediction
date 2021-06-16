from pipeline import run_pipeline
from monitoring.initialize_monitoring import initialize_monitoring
from config import configs_pipeline, v_dask
from local_conf import local_workers, local_threads_per_worker, local_memory_limit, only_base_conf_run
from tools.signature import GLOBAL_SIGNATURE

if __name__ == '__main__':
    initialize_monitoring()
    if v_dask:
        from dask.distributed import Client
        CLIENT = Client(n_workers=local_workers, threads_per_worker=local_threads_per_worker,
                        memory_limit=local_memory_limit)
        print(CLIENT.scheduler_info()['services'])
    for config in configs_pipeline:
        print(f'Pipeline runs with config: {config}')
        run_pipeline(config)
        GLOBAL_SIGNATURE.sig = ''

