import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.cm as cm

### Setting up directories
# project directory
proj_dir = os.getcwd()
# data/ directory where different data and all outputs are saved
data_dir = os.path.join(proj_dir, 'data')
# data/hein-daily/ directory for the original Hein-Daily database
orig_dir = os.path.join(data_dir, 'hein-daily')
plot_dir = os.path.join(data_dir, 'hein-daily-plot')
num_topics = 25

# Creating < >_10_bigrams.csv

# define an empty dataframe
df1 = pd.DataFrame()
for sess in range(97, 115):
    input_dir = os.path.join(data_dir, 'hein-daily-' + str(sess), 'clean')
    output_dir = os.path.join(data_dir, 'hein-daily-' + str(sess), 'tbip-fits', 'params')
    ideals = np.load(os.path.join(output_dir, 'ideal_point_loc.npy'))
    vocab = pd.read_csv(os.path.join(input_dir, 'vocabulary.txt'), header=None)
    # To be uncommented accordingly
    # for neutral_10_bigrams.csv
    beta = np.load(os.path.join(output_dir, 'neutral_topic_mean.npy'))   # neutral
    # for negative_10_bigrams.csv
    # beta = np.load(os.path.join(output_dir, 'negative_topic_mean.npy'))   # negative
    # for positve_10_bigrams.csv
    # beta = np.load(os.path.join(output_dir, 'positive_topic_mean.npy'))   # positive
    beta = np.exp(beta)   # model uses exp() values only
    t_quantile = np.quantile(ideals, 0.1)   # 10th quantile ideal point
    n_quantile = np.quantile(ideals, 0.9)   # 90th quantile ideal point
    for topic in range(0, 25):
        intensity = t_quantile * beta
        word_index = intensity[topic].argsort()[:10][::-1].tolist()   # n top words to include in wordcloud

        weights = []
        for i in word_index:
            weights.append(beta[topic][i])

        weights = np.exp(weights)
        weights = weights/np.sum(weights)
        words_n = []
        for i in word_index:
            words_n.append(vocab[0][i])
        dictionary = pd.Series(weights, index = words_n).to_dict()
        words_n.reverse()
        weights = np.flip(weights)
        df = pd.DataFrame()
        df['session'] = [sess] * 10
        df['topic'] = [topic + 1] * 10
        df['bigram'] = words_n
        df['weight'] = weights
        df1 = pd.concat([df1, df], axis=0)
        print("Session ", sess, " Topic ", topic+1, " done!")
# uncomment accordingly
df1.to_csv(os.path.join(plot_dir, 'neutral_10_bigrams.csv'), index=True)   # neutral
# df1.to_csv(os.path.join(plot_dir, 'negative_10_bigrams.csv'), index=True)   # negative
# df1.to_csv(os.path.join(plot_dir, 'positive_10_bigrams.csv'), index=True)   # positive
print("Saved!")

#print(df1)