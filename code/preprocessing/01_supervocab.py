import os
#import setup_utils as utils
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import defaultdict

# Data source : https://data.stanford.edu/congress_text#download-data
# Please download and unzip hein-daily.zip into orig_dir (described below)

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
min_df = 0.001                   # minimum document frequency
max_df = 0.3                     # maximum document frequency
stop_words = stopwords.tolist()  # stopwords
ngram_range = (2, 2)             # bigrams only
token_pattern = "[a-zA-Z]+"      # pattern

### Helper function
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



### Creating a complete vocabulary covering all the sessions

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

    # Remove senators who make less than 24 speeches
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

    count_vectorizer = CountVectorizer(min_df=min_df,
                                       max_df=max_df,
                                       stop_words=stop_words,
                                       ngram_range=ngram_range,
                                       token_pattern=token_pattern)

    # Learn initial document term matrix. This is only initial because we use it to
    # identify words to exclude based on author counts.
    counts = count_vectorizer.fit_transform(speeches.astype(str))
    vocabulary = np.array(
        [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(),
                                key=lambda kv: kv[1])])

    # Remove bigrams spoken by less than 10 Senators.
    counts_per_author = bincount_2d(author_indices, counts.toarray())
    author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
    acceptable_words = np.where(author_counts_per_word >= min_authors_per_word)[0]

    # Fit final document-term matrix with modified vocabulary.
    count_vectorizer = CountVectorizer(ngram_range=(2, 2),
                                       vocabulary=vocabulary[acceptable_words])
    counts = count_vectorizer.fit_transform(speeches.astype(str))
    vocabulary = np.array(
        [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(),
                                key=lambda kv: kv[1])])

    # counts_dense = remove_cooccurring_ngrams(counts, vocabulary) # not required since only bigrams are being considered
    # Remove speeches with no words.
    existing_speeches = np.where(np.sum(counts, axis=1) > 0)[0]
    counts = counts[existing_speeches]
    author_indices = author_indices[existing_speeches]
    # session specific vocabulary saved to ~/data/hein-daily/session-number/
    sess_dir = os.path.join(data_dir, 'hein-daily-' + str(i))
    if not os.path.exists(sess_dir):
        os.mkdir(sess_dir)
    np.savetxt(os.path.join(orig_dir, 'vocabulary_' + str(i) + '.txt'), vocabulary, fmt="%s")
    print("vocabulary saved for session "+str(i))

# pip install session_info
# import session_info
# session_info.show()
#
# create a combined vocabulary for all the sessions

super_vocab = [] # empty list

for i in range(97, 115):
    v = pd.read_csv(os.path.join(orig_dir, 'vocabulary_' + str(i) + '.txt'), header=None)
    v = v[0].tolist()
    super_vocab.append(v) #append session specific vocabulary

results_list = super_vocab #list of lists
results_union = set().union(*results_list) #set union of lists
vocab_full = list(results_union) #change datatype to list
vocab_full = sorted(vocab_full) #sorted alphabetically
#complete vocabulary saved to ~/data
np.savetxt(os.path.join(orig_dir, 'vocabulary.txt'), vocab_full, fmt="%s")

