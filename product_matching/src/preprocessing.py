import pandas as pd
import numpy as np
from scipy import stats
from joblib import load

def get_query(path):
    with open(path) as reader:
        line = reader.readline().strip()
    return line


def preprocess_features(line):
    columns_to_drop = ['6', '21', '25', '33', '44', '59', '65', '70']
    columns_to_drop.reverse()

    for col in columns_to_drop:
        del line[int(col)]

    scaler = load('../scaler.pkl')
    line = np.array(line).reshape(1, -1)
    return scaler.transform(line)
