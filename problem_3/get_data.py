import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import numpy as np

start = datetime(2017, 1, 1)
end = datetime(2021, 12, 31)
data = web.DataReader('AMZN', 'yahoo', start, end)
data.to_csv('data/AMZ_data.csv')