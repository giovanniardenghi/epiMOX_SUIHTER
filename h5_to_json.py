import pandas as pd
import sys

file_name = sys.argv[1]

res = pd.read_hdf(file_name)
res.to_json(file_name[:-2]+'json', date_format='iso')
