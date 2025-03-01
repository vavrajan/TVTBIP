
# Import global packages
import os
import time

import pandas as pd
from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import matplotlib.pyplot as plt
import shutil


flags.DEFINE_integer("seed", default=314159, help="Random seed to be used.")
flags.DEFINE_string("data_name", default='hein-daily', help="Name of the dataset to take outputs from.")
flags.DEFINE_string("simulation", default='simulation', help="Name of the newly created dataset.")
FLAGS = flags.FLAGS

### Define the scenarios for ideological positions
def get_ideal(scenario, author_party, ideal, s):
    match scenario:
        case "zero":
            return np.zeros(author_party.shape)
        case "party":
            return 0.5*(author_party == 'D') + 0.0*(author_party == 'I') - 0.5*(author_party == 'R')
        case "diverge":
            if s <= 100:
                return np.zeros(author_party.shape)
            else:
                return 0.05*(s-100) * (author_party == 'D') + 0.0 * (author_party == 'I') - 0.05*(s-100) * (author_party == 'R')
        case "estimate":
            return ideal

def main(argv):
    del argv
    print(tf.__version__)

    ### Setting up directories
    project_dir = os.getcwd()
    data_dir = os.path.join(project_dir, 'data')



    scenarios = ["zero", "party", "diverge", "estimate"]
    # scenarios = ["estimate"]
    # scenarios = "party"
    seed = FLAGS.seed

    # # Betas from the last session used for all sessions
    # neutral_topics = np.load(os.path.join(source_dir, '114', output, "neutral_topic_mean.npy"))
    # beta = neutral_topics
    # positive_topics = np.load(os.path.join(source_dir, '114', output, "positive_topic_mean.npy"))
    # negative_topics = np.load(os.path.join(source_dir, '114', output, "negative_topic_mean.npy"))
    # eta = 0.5 * (positive_topics - negative_topics) # todo try multiplying with 10

    for s in range(97, 115):
        print('Starting sampling session ' + str(s) + '.')
        source_dir = os.path.join(data_dir, FLAGS.data_name + '-' + str(s))
        ## Directory setup
        # original dataset we are trying to imitate
        in_dir = os.path.join(source_dir, 'clean')
        ou_dir = os.path.join(source_dir, 'tbip-fits', 'params')

        ## Load data
        # inputs
        author_indices = np.load(os.path.join(in_dir, "author_indices.npy")).astype(np.int32)
        author_data = np.loadtxt(os.path.join(in_dir, "author_map.txt"),
                                 dtype=str, delimiter=" ", usecols=[0, 1, -1])
        author_party = np.char.replace(author_data[:, 2], '(', '')
        author_party = np.char.replace(author_party, ')', '')
        author_map = np.char.add(author_data[:, 0], author_data[:, 1])
        # save them to input files for simulation
        # np.save(os.path.join(input_dir, "author_indices.npy"), author_indices)
        # np.savetxt(os.path.join(input_dir, "author_map.txt"), author_map, fmt="%s")

        # shutil.copy(os.path.join(in_dir, "author_map.txt"), os.path.join(input_dir, "author_map.txt"))
        # shutil.copy(os.path.join(in_dir, "vocabulary.txt"), os.path.join(input_dir, "vocabulary.txt"))

        # model parameters
        theta = tf.cast(tf.constant(pd.read_csv(os.path.join(ou_dir, "thetas.csv"), index_col=0).to_numpy()), "float32")
        neutral_topics = np.load(os.path.join(ou_dir, "neutral_topic_mean.npy"))
        beta = neutral_topics
        positive_topics = np.load(os.path.join(ou_dir, "positive_topic_mean.npy"))
        negative_topics = np.load(os.path.join(ou_dir, "negative_topic_mean.npy"))
        estimated_ideal = np.load(os.path.join(ou_dir, "ideal_point_loc.npy"))
        eta = 0.5 * (positive_topics - negative_topics)
        eta[eta < -1] = -1.0  # maybe try 2.0, but 1.0 seems fine
        eta[eta > 1] = 1.0

        ## Trigger different scenarios
        for scenario in scenarios:
            print('Scenario = ' + scenario)
            sim_dir = os.path.join(data_dir, FLAGS.simulation+'-'+scenario+'-'+str(s))
            if not os.path.exists(sim_dir):
                os.mkdir(sim_dir)
            # newly generated dataset
            input_dir = os.path.join(sim_dir, 'clean')
            if not os.path.exists(input_dir):
                os.mkdir(input_dir)

            # save/copy auxiliary files to clean directories for simulation
            np.save(os.path.join(input_dir, "author_indices.npy"), author_indices)
            shutil.copy(os.path.join(in_dir, "author_map.txt"), os.path.join(input_dir, "author_map.txt"))
            shutil.copy(os.path.join(in_dir, "vocabulary.txt"), os.path.join(input_dir, "vocabulary.txt"))

            # Create idealogical positions depending on scenario and session number s
            ideal = tf.cast(tf.constant(get_ideal(scenario, author_party, estimated_ideal, s)), "float32")
            # Get Poisson rates and sum them over topics
            rate = tf.math.reduce_sum(tf.math.exp(
                theta[:, :, tf.newaxis] + beta[tf.newaxis, :, :] +
                eta[tf.newaxis, :, :] * tf.gather(ideal, author_indices)[:, tf.newaxis, tf.newaxis]
            ), axis=1)
            # Create the Poisson distribution with given rates
            count_distribution = tfp.distributions.Poisson(rate=rate)
            # Sample the counts
            seed, sample_seed = tfp.random.split_seed(seed)
            counts = count_distribution.sample(seed=sample_seed)
            print(counts.shape)
            sparse_counts = sparse.csr_matrix(counts)
            print(sparse_counts.shape)
            sparse.save_npz(os.path.join(input_dir, "counts.npz"), sparse_counts)

        print('Session ' + str(s) + ' finished')


if __name__ == '__main__':
    app.run(main)
