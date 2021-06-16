import pandas as pd
import gzip
from flatten_json import flatten
import json
import re
import sys
from monitoring.time_it import timing


def flatton_json_data(filename):
    
    def unzip_and_fix_json(filename):
        with gzip.GzipFile(filename, "r") as f:
            data = f.read()
        data = data.decode('UTF-8')
        data = fix_json(data)
        data = json.loads(data)
        return data

    
    def flatten_json_file_into_df(data):
        dict_flattened = (flatten(record, '.') for record in data)
        df = pd.DataFrame(dict_flattened)
        return df

    
    def timestamp_converter(df):
        timestamp_cols = [col for col in df.columns if 'timestamp' in col]
        while len(timestamp_cols) != 0:
            element = timestamp_cols.pop()
            df[element] = pd.to_datetime(df[element], unit='ms')
            df[element] = df[element].astype('datetime64[s]')
        return df

    
    def fix_json(broken):
        parts = []
        start = 0
        try:
            idx = broken.index('}{', start)
            while idx != -1:
                parts.append(broken[start:idx + 1])
                start = idx + 1
                idx = broken.index('}{', start)
        except ValueError:
            pass
        parts.append(broken[start:])
        return ''.join(['[',','.join(map(lambda s: re.sub(r"(?<!\\)(?:\\{2})*\\x([0-9a-fA-F]{2})", '\\\\u00\\1', s), parts)),']'])

    if __name__ == '__main__':
        usage = 'Usage: fix-json.py [--validate] <input-file> <output-file>'
        if len(sys.argv) < 3 or len(sys.argv) > 4:
            print(usage, file=sys.stderr)
            sys.exit(1)
        validate = False
        off = 1
        if sys.argv[1].startswith('--'):
            off += 1
            if sys.argv[1] == '--validate':
                validate = True
            else:
                print(usage, file=sys.stderr)
                sys.exit(2)
        print('Loading...')
        with open(sys.argv[off], mode='r', encoding='utf-8') as inf:
            data = inf.read()
        print('Processing...')
        fixed = fix_json(data)

        with open(sys.argv[off + 1], mode='w', encoding='utf-8') as outf:
            outf.write(fixed)
        if validate:
            print('Validating...')
            with open(sys.argv[off + 1], mode='r', encoding='utf-8') as valf:
                json.load(valf)
            print('Validation successful')

    data = unzip_and_fix_json(filename)
    df = flatten_json_file_into_df(data)
    df = timestamp_converter(df)
    return df
