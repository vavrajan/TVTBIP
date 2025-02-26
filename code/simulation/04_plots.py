
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
from collections import defaultdict
from scipy.spatial import distance

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import matplotlib.axes as pax
import matplotlib.cm as cm
import seaborn as sns

import sys


### Setting up directories
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data', 'simulation')
all_dir = os.path.join(source_dir, 'all')
fig_dir = os.path.join(source_dir, 'fig')
output = 'output'

scenarios = ['_zero', '_party', '_diverge', '_estimate']

for scenario in scenarios:

    neutral_topics = defaultdict(list)   # dictionary style list
    positive_topics = defaultdict(list)
    negative_topics = defaultdict(list)

    for i in range(97, 115):
        tbip_path = os.path.join(source_dir, str(i), output)
        neutral_mean = np.load(os.path.join(tbip_path, 'neutral_topic_mean' + scenario + '.npy'))   # log values
        positive_mean = np.load(os.path.join(tbip_path, 'positive_topic_mean' + scenario + '.npy'))   # log values
        negative_mean = np.load(os.path.join(tbip_path, 'negative_topic_mean' + scenario + '.npy'))
        ideals = np.load(os.path.join(tbip_path, 'ideal_point_mean' + scenario + '.npy'))
        t_quantile = np.quantile(ideals, 0.1)
        n_quantile = np.quantile(ideals, 0.9)
        neutral_topics[i] = np.exp(neutral_mean)   # using neutral/negative/positive mean npy files only
        # positive_topics[i] = t_quantile * np.exp(neutral_mean)   # 10th quantile x beta
        # negative_topics[i] = n_quantile * np.exp(neutral_mean)   # 90th quantile x beta
        positive_topics[i] = np.exp(positive_mean)
        negative_topics[i] = np.exp(negative_mean)
        # all values with exp

    order = [1, 14, 19, 15, 20, 17, 6, 16, 7, 22, 25, 23,
             10, 4, 21, 18, 5, 13, 2, 8, 12, 9, 11, 24, 3]  # order of neutral topics
    topics = [
        'Trade', 'Export/Import', 'Commemoration', 'Supreme court', 'Finances',
        'Transport', 'Law enforcement', 'Middle East', 'Budget', 'Education',
        'Climate change', 'Taxes', 'Civil rights', 'Federal government', 'Health care',
        'Rhetoric', 'Wars', 'Nuclear arms', 'Social security', 'Army',
        'Natural resources', 'Human rights', 'Security', 'Puerto Rico', 'Party rhetoric']
    x_ticks = [str((1787+2*x) % 100).zfill(2)+'-'+str((1788+2*x) % 100).zfill(2)+'\n('+str(x)+')' for x in range(97, 115)]



    ### Neutral topics - no topic labels and no scale!
    # their order is to be used for the next plot
    sess_topics_cs = defaultdict(list)  # dictionary style list to store sessionwise cosine distances between topics

    for s in range(97, 114):
        topic_cs = np.empty([25, 25])  # empty numpy array to fill for each session, shape 25x25
        sess1 = s  # sessions
        sess2 = s + 1

        for i in range(0, 25):
            t1 = neutral_topics[sess1][i]  # topic 1
            for j in range(0, 25):
                t2 = neutral_topics[sess2][j]  # topic 2

                topic_cs[i][j] = 1 - distance.cosine(t1, t2)  # cosine similarity between topics 1 and 2
            sess_topics_cs[s].append(topic_cs[i][i])  # saving only the diagonal from the 25x25 array

    y = []
    for i in range(97, 114):
        y.append(sess_topics_cs[i])  # a list of lists, with all diagonal values

    arr_y = np.array(y)  # list to array, shape 17x25
    arr_y = np.transpose(arr_y)  # transpose, shape 25x17

    # sorting topics by row means to figure out which topic changes the most
    neu_topic_means = pd.DataFrame()  # empty datafram
    neu_topic_means['mean'] = np.mean(arr_y, axis=1)  # mean of each row
    neu_topic_means['topic'] = list(range(25))  # list 0-25
    neu_topic_means['label'] = topics
    neu_topic_means = neu_topic_means.sort_values('mean')  # sort as per mean of each row
    y_ticks = (neu_topic_means['topic'] + 1).tolist()  # y axis ticks as per sorted topics
    y_labs = neu_topic_means['label'].tolist()

    y2 = []
    for i in neu_topic_means['topic']:
        y2.append(arr_y[i])  # sorted rows from sorted dataframe

    arr_y2 = np.array(y2)  # sorted rows

    # With topic labels
    fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
    ax.grid(False)
    # ax.set_title('Topic', loc='left', pad=5)
    ax.set_xlabel('Session')
    # plt.title('Cosine similarities of Neutral topics')
    ax.set_xticks(range(0, len(x_ticks) * 1, 1))
    ax.set_xticklabels(x_ticks)
    ax.set_xlim(0, len(x_ticks)-1)
    ax.set_yticks([y+0.5 for y in range(0, len(y_ticks) * 1, 1)])
    ax.set_yticklabels(y_ticks)
    ax.set_ylim(0, len(y_ticks))
    im = plt.pcolormesh(arr_y2, cmap=cm.gray, edgecolors='white', linewidths=1,
                        antialiased=True)
    # plt.colorbar(im)
    # secax_y2 = ax.secondary_yaxis(location='left')
    # secax_y2.set_ylabel('')
    # secax_y2.set_yticks([y+0.5 for y in range(0, len(y_ticks) * 1, 1)])
    # secax_y2.set_yticklabels(y_labs, ha='left')
    # secax_y2.tick_params(axis='y', pad=120)

    plt.savefig(os.path.join(fig_dir, 'sorted_neutral_topics_cs_noscale' + scenario + '.pdf'), bbox_inches='tight')  # uncomment to save
    # plt.show()
    plt.close()
    # bright = closer, dark = farther





    ### Positive vs. Negative topics
    sess_topics_cs = defaultdict(list)  # dictionary style list to store sessionwise cosine distances between topics

    for s in range(97, 114):
        topic_cs = np.empty([25, 25])  # empty numpy array to fill for each session, shape 25x25
        for i in range(0, 25):
            t1 = positive_topics[s][i]  # topic 1
            t2 = negative_topics[s][i]  # topic 2
            sess_topics_cs[s].append(1 - distance.cosine(t1, t2))  #cosine similarity between +ve and -ve ideal topics of session

    y = []
    for i in range(97, 114):
        y.append(sess_topics_cs[i])  # a list of lists, with all diagonal values

    arr_y = np.array(y)  # list to array, shape 17x25
    arr_y = np.transpose(arr_y)  # transpose, shape 25x17

    # sorting topics by row means to figure out which topic changes the most
    y_ticks = (neu_topic_means['topic'] + 1).tolist()  # y axis ticks as per sorted topics
    y_labs = neu_topic_means['label'].tolist()

    y2 = []
    for i in neu_topic_means['topic']:
        y2.append(arr_y[i])  # sorted rows from sorted dataframe

    arr_y2 = np.array(y2)  # sorted rows

    # With topic labels
    fig, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax.grid(False)
    ax.set_title('Topic', loc='left', pad=5)
    ax.set_xlabel('Session')
    # plt.title('Cosine similarities of Neutral topics')
    ax.set_xticks(range(0, len(x_ticks) * 1, 1))
    ax.set_xticklabels(x_ticks)
    ax.set_xlim(0, len(x_ticks)-1)
    ax.set_yticks([y+0.5 for y in range(0, len(y_ticks) * 1, 1)])
    ax.set_yticklabels(y_ticks)
    ax.set_ylim(0, len(y_ticks))
    im = plt.pcolormesh(arr_y2, cmap=cm.gray, edgecolors='white', linewidths=1,
                        antialiased=True)
    plt.colorbar(im)
    secax_y2 = ax.secondary_yaxis(location='left')
    secax_y2.set_ylabel('')
    secax_y2.set_yticks([y+0.5 for y in range(0, len(y_ticks) * 1, 1)])
    secax_y2.set_yticklabels(y_labs, ha='left')
    secax_y2.tick_params(axis='y', pad=120)

    plt.savefig(os.path.join(fig_dir, 'pos_vs_neg_topics_cs_sorted_by_neu' + scenario + '.pdf'),
                bbox_inches='tight')  # uncomment to save
    # plt.show()
    plt.close()
    # bright = closer, dark = farther
