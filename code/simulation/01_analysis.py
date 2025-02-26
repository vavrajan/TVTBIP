import functools
import time
import os
import pandas as pd
import csv

# import tensorflow_version 1.15
import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import non_negative_factorization
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app
from absl import flags

import sys

### Setting up directories
project_dir = os.getcwd()
data_name = 'simulation'
source_dir = os.path.join(project_dir, 'data', data_name)
all_dir = os.path.join(source_dir, 'all')
if not os.path.exists(all_dir):
    os.mkdir(all_dir)



num_topics = 25
output = 'output'
scenarios = ['_zero', '_party', '_diverge', '_estimate']
# scenarios = ['_zero']
# scenarios = ['_party']
# scenarios = ['_party', '_diverge', '_estimate']
max_sess = 115

for scenario in scenarios:
    for sess in range(97, max_sess):
        save_dir = os.path.join(source_dir, str(sess), output)
        ideal_point_mean = np.load(os.path.join(save_dir, "ideal_point_mean" + scenario + ".npy"))

        # author mapping to ideal point
        author_map = np.loadtxt(os.path.join(source_dir, str(sess), 'input', 'author_map.txt'),
                                dtype=str,
                                delimiter='\n',
                                comments='//')

        speaker_IP = pd.DataFrame(columns=['speaker', 'ideal_point'])  # create an empty dataframe
        speaker_IP['speaker'] = author_map
        speaker_IP['ideal_point'] = ideal_point_mean
        speaker_IP.to_csv(os.path.join(save_dir, "ideal_point_speakers" + scenario + ".csv"), header=True)



for scenario in scenarios:
    ip = pd.read_csv(os.path.join(source_dir, str(97), output, "ideal_point_speakers" + scenario + ".csv"))
    ip.columns = ['Index', 'speaker', 'ideal_point_97']
    # ip = ip.drop(columns=ip.columns[0], axis=1, inplace=True)

    for sess in range(98, max_sess):
        df = pd.read_csv(os.path.join(source_dir, str(sess), output, "ideal_point_speakers" + scenario + ".csv"))
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        df.columns = ['speaker', 'ideal_point_'+str(sess)]
        print(df.shape)
        ip = df.merge(ip, on='speaker', how='outer')
    # print(ip.columns)
    ip = ip.drop(['Index'], axis=1)
    ip = ip[ip.columns[::-1]]
    cols = list(ip.columns)
    cols = [cols[-1]] + cols[:-1]
    ip = ip[cols]
    ip.to_csv(os.path.join(all_dir, "ideal_points_all_sessions" + scenario + ".csv"), index=False)


