import os
import numpy as np
import sys
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import defaultdict

import analysis_utils as utils

### Setting up directories
# project directory
proj_dir = os.getcwd()
# data/hein-daily/ directory where Hein-Daily database and all outputs are saved
data_dir = os.path.join(proj_dir, 'data', 'hein-daily')
# data/hein-daily/orig/ directory for the original Hein-Daily database
orig_dir = os.path.join(data_dir, 'orig')
plot_dir = os.path.join(data_dir, 'plot')
num_topics = 25
output = 'output'
i = 107
param_dir = os.path.join(data_dir, str(i), output)


# this codejunk is only to create raw_documents.txt

### Stopwords
# predefined set of stopwords saved in "stopwords.txt"
stopwords = np.loadtxt(os.path.join(data_dir, "stopwords.txt"), dtype=str)
# stopwords available at: https://github.com/keyonvafa/tbip/blob/master/setup/stopwords/senate_speeches.txt
# to be downloaded and saved to data_dir as defined above

### Parameters
min_speeches = 24            # minimum number of speeches given by a senator
min_authors_per_word = 10    # minimum number of senators using a bigram

### Parameters for CountVectorizer
min_df = 0.001                   # minimum document frequency
max_df = 0.3                     # maximum document frequency
stop_words = stopwords.tolist()  # stopwords
ngram_range = (2, 2)             # bigrams only
token_pattern = "[a-zA-Z]+"      # pattern
vocab = pd.read_csv(os.path.join(data_dir, 'vocabulary.txt'), header=None)  # path to complete vocabulary
vocabulary = vocab[0].tolist()   # vocabulary as a list


speeches = pd.read_csv(os.path.join(orig_dir, 'speeches_' + str(i) + '.txt'),
                       encoding="ISO-8859-1", sep="|",
                       error_bad_lines = False)  # on_bad_lines='warn')
description = pd.read_csv(os.path.join(orig_dir, 'descr_' + str(i) + '.txt'),
                          encoding="ISO-8859-1", sep="|")
speaker_map = pd.read_csv(os.path.join(orig_dir, str(i) + '_SpeakerMap.txt'),
                          encoding="ISO-8859-1", sep="|")

merged_df = speeches.merge(description, left_on='speech_id', right_on='speech_id')
df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

# Only look at senate speeches.
# to select speakers with speeches in the senate (includes Senators and House Reps)
senate_df = df[df['chamber_x'] == 'S']
# to select ONLY Senators uncomment the next line
# senate_df = df[df['chamber_y'] == 'S'] ##  here 7.2
speaker = np.array(
    [' '.join([first, last]) for first, last in
     list(zip(np.array(senate_df['firstname']),
              np.array(senate_df['lastname'])))])
speeches = np.array(senate_df['speech'])
party = np.array(senate_df['party'])
# Remove senators who make less than 24 speeches
unique_speaker, speaker_counts = np.unique(speaker, return_counts=True)
absent_speakers = unique_speaker[np.where(speaker_counts < min_speeches)]
absent_speaker_inds = [ind for ind, x in enumerate(speaker)
                           if x in absent_speakers]
speaker = np.delete(speaker, absent_speaker_inds)
speeches = np.delete(speeches, absent_speaker_inds)

party = np.delete(party, absent_speaker_inds)
speaker_party = np.array([speaker[i] + " (" + party[i] + ")" for i in range(len(speaker))])

# Create mapping between names and IDs.
speaker_to_speaker_id = dict(
        [(y.title(), x) for x, y in enumerate(sorted(set(speaker_party)))])
author_indices = np.array(
        [speaker_to_speaker_id[s.title()] for s in speaker_party])
author_map = np.array(list(speaker_to_speaker_id.keys()))

# count_vectorizer = CountVectorizer(min_df,
#                                    max_df,
#                                    stop_words,
#                                    ngram_range,
#                                    token_pattern)

# counts = count_vectorizer.fit_transform(speeches.astype(str))

# Fit final document-term matrix with complete vocabulary.
# count_vectorizer = CountVectorizer(ngram_range=(2, 2), vocabulary=vocabulary)
# counts = count_vectorizer.fit_transform(speeches.astype(str))


# print(speeches.shape)
# print(speeches[1000])
# s2 = speeches.tolist()
# print(s2[1000])
# print(len(s2))



# file = open(os.path.join(data_dir, "input", "raw_documents.txt"), "w")
# for element in s2:
#     file.write(element + "\n")
# file.close()

## create DTM
count_vectorizer = CountVectorizer(ngram_range=(2, 2), vocabulary=vocabulary)
s2 = speeches.tolist()
counts = count_vectorizer.fit_transform(s2)

#Remove speeches with no words.
existing_speeches = np.where(np.sum(counts, axis=1) > 0)[0]

summe = counts.sum(axis=1)
print(existing_speeches)
counts = counts[existing_speeches]
speeches2 = np.array(s2)[existing_speeches]
speeches2.shape

file = open(os.path.join(data_dir, str(i), "input", "raw_documents.txt"), "w")
for element in speeches2:
    file.write(element + "\n")
file.close()

summe = np.where(counts.sum(axis=1) == 0)[0]
summe = counts.sum(axis=1)

# Load TBIP data.
data_dir2 = os.path.join(data_dir, str(i), "input")
(counts, vocabulary, author_indices,
 author_map, raw_documents) = utils.load_text_data(data_dir2)


# Load TBIP parameters.
document_mean = np.load(os.path.join(param_dir, "document_topic_mean.npy"))
objective_topic_loc = np.load(os.path.join(param_dir, "objective_topic_loc.npy"))
objective_topic_scale = np.load(os.path.join(param_dir, "objective_topic_scale.npy"))
ideological_topic_loc = np.load(os.path.join(param_dir, "ideological_topic_loc.npy"))
ideological_topic_scale = np.load(os.path.join(param_dir, "ideological_topic_scale.npy"))
ideal_point_mean = np.load(os.path.join(param_dir, "ideal_point_mean.npy"))

objective_topic_mean = np.exp(objective_topic_loc +
                              objective_topic_scale ** 2 / 2)

ideological_topic_mean = ideological_topic_loc

raw_documents.shape

# Find influential speeches for Bernie Sanders' ideal point.
verbosity_weights = utils.get_verbosity_weights(counts, author_indices)
smith_top_indices, smith_top_words = utils.compute_likelihood_ratio(
    "Gordon Smith (R)",
    ideal_point_mean,
    counts,
    vocabulary,
    author_indices,
    author_map,
    document_mean,
    objective_topic_mean,
    ideological_topic_mean,
    verbosity_weights,
    null_ideal_point=0.,
    log_counts=True)
smith_top_speeches = speeches2[smith_top_indices]

author_indices.shape

smith_top_indices

speeches2.shape

speeches2

speeches2.shape

author_map

author_index = np.where(author_map == 'Gordon Smith (R)')[0][0]
author_documents = np.where(author_indices == author_index)[0]

author_index

author_map[31]

author_documents

speeches2[862]

speeches2[1493]

speeches2[2838]

speeches2[33452]

senate_df

