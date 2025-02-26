import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
from collections import defaultdict
import seaborn as sns
from scipy.spatial import distance

# The aim of this file is to process all the estimated data and compute (and save) everything needed for plots.

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

### store beta, eta, theta and ideal points

for sess in range(97, 115):
    save_dir = os.path.join(data_dir, str(sess), output)
    topics = []
    bigram_index = []
    objective_topic_loc = np.load(os.path.join(save_dir, "objective_topic_loc.npy"))
    document_topic_mean = np.load(os.path.join(save_dir, "document_topic_mean.npy"))
    itl = np.load(os.path.join(save_dir, "ideological_topic_loc.npy"))
    ideal_point_mean = np.load(os.path.join(save_dir, "ideal_point_mean.npy"))
    beta = objective_topic_loc
    vocab = pd.read_csv(os.path.join(data_dir, 'vocabulary.txt'), header=None)

    for k in range(0, num_topics):
        arr = beta[k]
        list_grams = arr.argsort()[-len(vocab):][::-1]   # sorting vocabulary by significance to topic
        terms = []
        new_index = []

        for j in list_grams:
            if ' ' in vocab[0][j]:          # selecting only bigrams from the vocabulary
                terms.append(vocab[0][j])   # list of bigrams only
                new_index.append(j)         # index of bigrams only
        bigram_index.append(new_index)      # index of sorted bigrams for each topic
        topics.append(terms)                # bigrams assigned to each topic

    df = pd.DataFrame()   # adding everything to a dataframe
    df['topic'] = [0] * len(bigram_index[0])   # topic 1
    df['bigram'] = bigram_index[0]

    betas = []
    for i in bigram_index[0]:
        betas.append(np.exp(beta[0][i]))

    df['beta'] = betas

    for t in range(1, num_topics):   # adding for topics 1-24
        df_1 = pd.DataFrame()   # appending dataframe
        df_1['topic'] = [t] * len(bigram_index[t])
        df_1['bigram'] = bigram_index[t]

        betas = []
        for i in bigram_index[t]:
            betas.append(np.exp(beta[t][i]))

        df_1['beta'] = betas

        df = pd.concat([df, df_1], axis=0)   # create a dataframe for all the topics

    #print(df)
    df.to_csv(os.path.join(save_dir, 'betas.csv'), index=True)

    #thetas
    theta_df = document_topic_mean
    pd.DataFrame(theta_df).to_csv(os.path.join(save_dir, 'thetas.csv'), index=True)

    #eta
    bigram_index = list(range(itl.shape[1]))   # create len(num_bigrams) columns
    eta_df = pd.DataFrame(columns = bigram_index)
    print(itl.shape[0])
    for j in range(0, itl.shape[0]):
        to_append = itl[j].tolist()
        a_series = pd.Series(to_append, index=eta_df.columns)   # list to a series indexed by columns
        eta_df = eta_df.append(a_series, ignore_index=True)
    eta_df.to_csv(os.path.join(save_dir, 'etas.csv'), index=True)

    # author mapping to ideal point
    author_map = np.loadtxt(os.path.join(data_dir, str(sess), 'input', 'author_map.txt'),
                        dtype=str,
                        delimiter='\n',
                        comments='//')

    speaker_IP = pd.DataFrame(columns=['speaker', 'ideal_point'])   # create an empty dataframe
    speaker_IP['speaker'] = author_map
    speaker_IP['ideal_point'] = ideal_point_mean
    speaker_IP
    speaker_IP.to_csv(os.path.join(save_dir, "ideal_point_speakers.csv"), header=True)

# speeches for particular speakers with date

speakers = ['BYRON DORGAN (D)', 'DALE BUMPERS (D)', 'THOMAS HARKIN (D)', 'CHRISTOPHER MURPHY (D)', 'PAUL WELLSTONE (D)',
            'DANIEL INOUYE (D)', 'ROBERT TORRICELLI (D)', 'BEN NELSON (D)', 'ARLEN SPECTER (D)', 'EVAN BAYH (D)',
            'JESSE HELMS (R)', 'GORDON SMITH (R)', 'JOHN BARRASSO (R)', 'JOHN HOEVEN (R)', 'MIKE JOHANNS (R)',
            'ED BRYANT (R)', 'JOHN CHAFEE (R)', 'DIRK KEMPTHORNE (R)', 'PHIL GRAMM (R)', 'HOWARD MCKEON (R)',
            'JAMES JEFFORDS (I)', 'BERNARD SANDERS (I)', 'JOSEPH LIEBERMAN (I)', 'ANGUS KING (I)']

for counter in range(0, len(speakers)):
    df_s = pd.DataFrame(columns=['date', 'speech'])
    for i in range(97, 115):
        if i < 100:
            session = '0' + str(i)
        else:
            session = str(i)
        speeches = pd.read_csv(os.path.join(orig_dir, 'speeches_' + session + '.txt'),
                               encoding="ISO-8859-1", sep="|",  # quoting=3, # without quoting
                               error_bad_lines=False)
        description = pd.read_csv(os.path.join(orig_dir, 'descr_' + session + '.txt'),
                                  encoding="ISO-8859-1", sep="|")
        speaker_map = pd.read_csv(os.path.join(orig_dir, session + '_SpeakerMap.txt'),
                                  encoding="ISO-8859-1", sep="|")

        merged_df = speeches.merge(description, left_on='speech_id', right_on='speech_id')
        df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

        # Only look at senate speeches.
        senate_df = df[df['chamber_x'] == 'S']
        #senate_df = senate_df.groupby(['chamber_y']).get_group('S')
        speaker = np.array(
            [' '.join([first, last]) for first, last in
             list(zip(np.array(senate_df['firstname']),
                      np.array(senate_df['lastname'])))])
        senate_df['speaker'] = speaker
        speeches = np.array(senate_df['speech'])
        party = np.array(senate_df['party'])
        speaker_party = np.array([speaker[i] + " (" + party[i] + ")" for i in range(len(speaker))])
        senate_df['speaker_party'] = speaker_party
        df_a = pd.DataFrame(columns = ['date', 'speech'])
        try:
            df_a['date'] = senate_df.groupby(['speaker_party']).get_group(speakers[counter])['date']
            df_a['speech'] = senate_df.groupby(['speaker_party']).get_group(speakers[counter])['speech']
        except KeyError:
            continue
        df_s = df_s.append(df_a, ignore_index = True)
    df_s.to_csv(os.path.join(save_dir, speakers[counter].split(" ")[-2]+'.csv'), index=False)
    print('done '+str(counter))

#  number of speeches per speaker per session

sp_s = defaultdict(list)
sp_p = defaultdict(list)

for i in range(97, 115):
    if i < 100:
        session = '0' + str(i)
    else:
        session = str(i)
    speeches = pd.read_csv(os.path.join(orig_dir, 'speeches_' + session + '.txt'),
                           encoding="ISO-8859-1", sep="|",  # quoting=3, # without quoting
                           error_bad_lines=False) # on_bad_lines='warn')
    description = pd.read_csv(os.path.join(orig_dir, 'descr_' + session + '.txt'),
                              encoding="ISO-8859-1", sep="|")
    speaker_map = pd.read_csv(os.path.join(orig_dir, session + '_SpeakerMap.txt'),
                              encoding="ISO-8859-1", sep="|")

    merged_df = speeches.merge(description, left_on='speech_id', right_on='speech_id')
    df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

    # Only look at senate speeches.
    senate_df = df[df['chamber_x'] == 'S']
    # senate_df = senate_df.groupby(['chamber_y']).get_group('S')
    speaker = np.array(
        [' '.join([first, last]) for first, last in
         list(zip(np.array(senate_df['firstname']),
                  np.array(senate_df['lastname'])))])
    party = np.array(senate_df['party'])
    senate_df['speaker'] = speaker
    speaker_party = np.array([speaker[i] + " (" + party[i] + ")" for i in range(len(speaker))])
    senate_df['speaker_party'] = speaker_party
    u_spk = np.unique(speaker_party)
    sp_p[i] = np.unique(speaker_party)
    # speeches = np.array(senate_df['speech'])
    # party = np.array(senate_df['party'])
    for speakers in u_spk:
        sp_s[i].append(len(senate_df.groupby(['speaker_party']).get_group(speakers)))

c = ['speaker', 'speeches_97']
df1 = pd.DataFrame(columns=c)
df1['speaker'] = sp_p[97]
df1['speeches_97'] = sp_s[97]

for sess in range(98, 115):
    df_append = pd.DataFrame(columns=['speaker', 'speeches_'+str(sess)]) #new dataframe per session to append
    df_append['speaker'] = sp_p[sess]
    df_append['speeches_'+str(sess)] = sp_s[sess]
    #append the dataset
    df1 = pd.concat([df1.set_index('speaker'),df_append.set_index('speaker')], axis=1, join='outer').reset_index()
    df1.rename(columns={'index':'speaker'}, inplace=True)

# save as csv
# prior to preprocessing
df1.to_csv(os.path.join(plot_dir, 'speeches_by_speakers.csv'), index=False)

##PH commented this
#ip = pd.DataFrame()
#for sess in range(97, 115):
#    filename = os.path.join(data_dir, str(sess), output, 'ideal_point_speakers.csv')
#    df = pd.read_csv(filename)
    #last_col =
#    ip = ip.join(df[last_col], how='outer')
#ip.reset_index(inplace=True)
#ip.to_csv(os.path.join(plot_dir,'test.csv'), index=False)

#after preprocessing
#ip = pd.read_csv(os.path.join(path, 'ideal_points_all_sessions.csv'))
#senators = ip['senator'].tolist()
#for i in range(0, len(senators)):
#    senators[i] = senators[i].upper()

#new_df = df_1[df_1.speaker.isin(senators)]
#new_df.to_csv(os.path.join(plot_dir, 'speeches_by_preprocessed_speakers.csv', index = False))

ip = pd.read_csv(os.path.join(data_dir, str(97), output, 'ideal_point_speakers.csv'))
ip.columns = ['Index', 'speaker', 97]
#ip = ip.drop(columns=ip.columns[0], axis=1, inplace=True)

for sess in range(98, 115):
    df = pd.read_csv(os.path.join(data_dir, str(sess), output, 'ideal_point_speakers.csv'))
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df.columns = ['speaker', sess]
    print(df.shape)
    ip=df.merge(ip,on ='speaker', how='outer')
# print(ip.columns)
ip = ip.drop(['Index'], axis=1)
ip=ip[ip.columns[::-1]]
cols = list(ip.columns)
cols = [cols[-1]] + cols[:-1]
ip = ip[cols]
ip.to_csv(os.path.join(plot_dir, 'ideal_points_all_sessions.csv'), index=False)


ip = pd.read_csv(os.path.join(plot_dir, 'ideal_points_all_sessions.csv'))
senators = ip['speaker'].tolist()
for i in range(0, len(senators)):
    senators[i] = senators[i].upper()

new_df = df1[df1.speaker.isin(senators)]
new_df.to_csv(os.path.join(plot_dir, 'speeches_by_preprocessed_speakers.csv'), index = False)

# cosine similarities between positive and negative topics

positive_topics = defaultdict(list)
negative_topics = defaultdict(list)

for i in range(97, 115):
    tbip_path = os.path.join(data_dir, str(i), output)   # tbip directory with output files
    positive_mean = np.load(os.path.join(tbip_path, 'positive_topic_mean.npy'))
    negative_mean = np.load(os.path.join(tbip_path, 'negative_topic_mean.npy'))
    positive_topics[i] = np.exp(positive_mean)
    negative_topics[i] = np.exp(negative_mean)
    # all values with exp

cs_pn = pd.DataFrame(columns=['topic']) # dataframe to save

for sess in range(97, 115):
    similarity = []
    for topic in range(0, 25):
        t1 = positive_topics[sess][topic]
        t2 = negative_topics[sess][topic]
        similarity.append(1 - distance.cosine(t1, t2))
    cs_pn[sess] = similarity

cs_pn['topic'] = list(range(1, 26))
cs_pn.to_csv(os.path.join(plot_dir, 'posneg_cs.csv'), index=False)