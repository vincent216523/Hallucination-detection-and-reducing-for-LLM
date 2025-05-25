#
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.data_utils import read_pickle

dir = os.getcwd()
results_dir = os.path.join(dir, 'result')
results = []
target_index = '500-1500'

def get_results(results_dir, target_index):
    """
    Get results from the result directory
    """
    results = []

    for file in os.listdir(results_dir):
        if file.endswith('.pkl') and target_index in file:
            file_path = os.path.join(results_dir, file)
            data = read_pickle(file_path)
            results.append(data)

    dataset = []
    for q in data:
        question = q[0]['question']
        for dict in q:
            row_val = []
            row_val.append(question)
            if dict is not None:
                for key in dict:
                        if key == 'correct':
                            row_val.append(dict[key][0])
                        else:
                            row_val.append(dict[key])
                dataset.append(row_val)

    columns = data[0][0].keys()
    columns_list = ['trivia_qa']
    columns_list.extend(list(columns))
    df = pd.DataFrame(dataset,columns = columns_list)
    return df
