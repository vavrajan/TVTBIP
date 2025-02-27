from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
data_dir = os.path.join(project_dir, 'data')
orig_dir = os.path.join(project_dir, 'data', 'hein-daily')
sim_dir = os.path.join(project_dir, 'data', 'simulation')
if not os.path.exists(sim_dir):
    os.mkdir(sim_dir)
# data = 'hein-daily'
# data = 'simulation-party'
# data = 'simulation-zero'
# data = 'simulation-estimate'
# data = 'simulation-diverge'
max_sess = 115

# Ideal points for each session separately to csv
for data in ['simulation-zero', 'simulation-party', 'simulation-diverge', 'simulation-estimate']:
    for s in range(97, max_sess):
        data_name = data+'-'+str(s)
        print("Data and session: " + data_name)
        source_dir = os.path.join(data_dir, data_name)
        clean_dir = os.path.join(source_dir, 'clean')
        tbip_dir = os.path.join(source_dir, 'tbip-fits', 'params')
        input_dir = os.path.join(orig_dir, str(s), 'input')

        # simulated data
        ideal_point_loc = np.load(os.path.join(tbip_dir, "ideal_point_loc.npy"))
        eta_loc = np.load(os.path.join(tbip_dir, "ideological_topic_loc.npy"))
        # sim_eta_scl = np.std(eta_loc)
        sim_eta_scl = np.quantile(eta_loc, 0.75) - np.quantile(eta_loc, 0.25)

        # orig data
        positive_topics = np.load(os.path.join(input_dir, "positive_topic_mean.npy"))
        negative_topics = np.load(os.path.join(input_dir, "negative_topic_mean.npy"))
        eta = 0.5 * (positive_topics - negative_topics)
        # orig_eta_scl = np.std(eta)
        orig_eta_scl = np.quantile(eta, 0.75) - np.quantile(eta, 0.25)

        # scale coefficient
        scale_coefficient = sim_eta_scl / orig_eta_scl
        # print("Scale coefficient = ", np.add(scale_coefficient, 0.0))
        # print("Scale coefficient = {:.3f}".format(np.add(scale_coefficient, 0.0)))
        print("Scale coefficient = {:.3f}".format(scale_coefficient))
        # print("Scale coefficient = " + str(scale_coefficient))

        # author mapping to ideal point
        author_map = np.loadtxt(os.path.join(clean_dir, 'author_map.txt'),
                                dtype=str,
                                delimiter='\n',
                                comments='//')

        speaker_IP = pd.DataFrame(columns=['speaker', 'ideal_point'])  # create an empty dataframe
        speaker_IP['speaker'] = author_map
        speaker_IP['ideal_point'] = ideal_point_loc * scale_coefficient
        speaker_IP.to_csv(os.path.join(tbip_dir, "ideal_point_speakers_rescaledIQR.csv"), header=True)

    # Ideal points from all seesions together to one csv
    ip = pd.read_csv(os.path.join(data_dir, data+'-97', 'tbip-fits', 'params',
                                  "ideal_point_speakers_rescaledIQR.csv"))
    ip.columns = ['Index', 'speaker', 'ideal_point_97']
    # ip = ip.drop(columns=ip.columns[0], axis=1, inplace=True)
    for sess in range(98, max_sess):
        df = pd.read_csv(os.path.join(data_dir, data+'-'+str(sess), 'tbip-fits', 'params',
                                      "ideal_point_speakers_rescaledIQR.csv"))
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        df.columns = ['speaker', 'ideal_point_' + str(sess)]
        print(df.shape)
        ip = df.merge(ip, on='speaker', how='outer')
    # print(ip.columns)
    ip = ip.drop(['Index'], axis=1)
    ip = ip[ip.columns[::-1]]
    cols = list(ip.columns)
    cols = [cols[-1]] + cols[:-1]
    ip = ip[cols]
    ip.to_csv(os.path.join(sim_dir, data+"_ideal_points_all_sessions_rescaledIQR.csv"), index=False)