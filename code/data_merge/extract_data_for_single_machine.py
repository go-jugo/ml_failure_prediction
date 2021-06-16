import pandas as pd
from monitoring.time_it import timing


def extract_data_for_single_machine(df, machine_id='MAZAK-M7303290458'):
    valid = ['MAZAK-M7303290458', 'Integrex', 'VTC530C']
    if machine_id not in valid:
        raise ValueError('machine_id is not a valid id. Choose id from list=[\'MAZAK-M7303290458\', \'Integrex\', \'VTC530C\']')
    df_machine = df.loc[df['machineId'] == machine_id]
    if len(df_machine) > 0:
        return df_machine
    else:
        df_machine = pd.DataFrame()
        return df_machine