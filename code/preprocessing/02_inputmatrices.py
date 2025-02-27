import os
# import setup_utils as utils
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import defaultdict

# !!! run '01_supervocab.py' first to create vocabulary common to all sessions

### Setting up directories
# project directory
proj_dir = os.getcwd()
# data/ directory where different data and all outputs are saved
data_dir = os.path.join(proj_dir, 'data')
# data/hein-daily/ directory for the original Hein-Daily database
orig_dir = os.path.join(data_dir, 'hein-daily')

### Stopwords
# predefined set of stopwords saved in "stopwords.txt"
stopwords = np.loadtxt(os.path.join(orig_dir, "stopwords.txt"), dtype=str)
# stopwords available at: https://github.com/keyonvafa/tbip/blob/master/setup/stopwords/senate_speeches.txt
# to be downloaded and saved to data_dir as defined above

### Parameters
min_speeches = 24            # minimum number of speeches given by a senator
min_authors_per_word = 10    # minimum number of senators using a bigram

### Parameters for CountVectorizer
# Commented ones were used for initial creation of total vocabulary, no longer needed
# min_df = 0.001                   # minimum document frequency
# max_df = 0.3                     # maximum document frequency
# stop_words = stopwords.tolist()  # stopwords
# ngram_range = (2, 2)             # bigrams only
# token_pattern = "[a-zA-Z]+"      # pattern
# Now it is sufficient to load only the vocabulary
vocab = pd.read_csv(os.path.join(orig_dir, 'vocabulary.txt'), header=None)  # path to complete vocabulary
vocabulary = vocab[0].tolist()   # vocabulary as a list

# Helper function
# source code originally available at: https://github.com/keyonvafa/tbip
# Count number of occurrences of each value in array of non-negative integers
# documentation: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

def bincount_2d(x, weights):
    _, num_topics = weights.shape
    num_cases = np.max(x) + 1
    counts = np.array(
      [np.bincount(x, weights=weights[:, topic], minlength=num_cases)
       for topic in range(num_topics)])
    return counts.T

for i in range(97, 115):
    if i < 100:
        session = '0'+str(i)
    else:
        session = str(i)
    speeches = pd.read_csv(os.path.join(orig_dir, 'speeches_' + session + '.txt'),
                           encoding="ISO-8859-1", sep="|",  # quoting=3, # without quoting
                           error_bad_lines = False)
                           # on_bad_lines='warn')
    description = pd.read_csv(os.path.join(orig_dir, 'descr_' + session + '.txt'),
                              encoding="ISO-8859-1", sep="|")
    speaker_map = pd.read_csv(os.path.join(orig_dir, session + '_SpeakerMap.txt'),
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

    # Remove senators who make less than 24 speeches.
    unique_speaker, speaker_counts = np.unique(speaker, return_counts=True)
    absent_speakers = unique_speaker[np.where(speaker_counts < min_speeches)]
    absent_speaker_inds = [ind for ind, x in enumerate(speaker) if x in absent_speakers]
    speaker = np.delete(speaker, absent_speaker_inds)
    speeches = np.delete(speeches, absent_speaker_inds)
    party = np.delete(party, absent_speaker_inds)
    speaker_party = np.array([speaker[i] + " (" + party[i] + ")" for i in range(len(speaker))])

    # Create mapping between names and IDs.
    speaker_to_speaker_id = dict([(y.title(), x) for x, y in enumerate(sorted(set(speaker_party)))])
    author_indices = np.array([speaker_to_speaker_id[s.title()] for s in speaker_party])
    author_map = np.array(list(speaker_to_speaker_id.keys()))

    # Fit final document-term matrix with complete vocabulary.
    count_vectorizer = CountVectorizer(ngram_range=(2, 2), vocabulary=vocabulary)
    counts = count_vectorizer.fit_transform(speeches.astype(str))

    # Remove speeches with no words.
    existing_speeches = np.where(np.sum(counts, axis=1) > 0)[0]
    counts = counts[existing_speeches]
    author_indices = author_indices[existing_speeches]

    input_dir = os.path.join(data_dir, 'hein-daily-' + str(i), 'clean')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    #saving input matrices for TV-TBIP
    sparse.save_npz(os.path.join(input_dir, 'counts.npz'),
                    sparse.csr_matrix(counts).astype(np.float32))
    np.save(os.path.join(input_dir, "author_indices.npy"), author_indices)
    np.savetxt(os.path.join(input_dir, "author_map.txt"), author_map, fmt="%s")
    np.savetxt(os.path.join(input_dir, "vocabulary.txt"), vocabulary, fmt="%s")

    print('done for session ' + str(i))

