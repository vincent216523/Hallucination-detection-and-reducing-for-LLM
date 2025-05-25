import pickle
import os

from utils.data_utils import read_pickle
import argparse
import numpy as np
import scipy as sp

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# init cwd
cwd = os.path.dirname(__file__)
results_dir = os.path.join(cwd, 'result')




'''
Dict values:
'question', 
'response', 
'str_response', 
'logits', 
'start_pos', 
'correct', 
'first_fully_connected', 
'final_fully_connected', 
'first_attention', 
'final_attention', 
'attributes_first', 
'turn', 
'idx', 
'hallucination'
'''