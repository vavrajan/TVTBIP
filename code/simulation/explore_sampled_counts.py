
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

flags.DEFINE_integer("seed", default=314159, help="Random seed to be used.")
flags.DEFINE_string("data_name", default='hein-daily', help="Name of the dataset to take outputs from.")
flags.DEFINE_string("simulation", default='simulation', help="Name of the newly created dataset.")
FLAGS = flags.FLAGS


def main(argv):
    del argv
    print(tf.__version__)

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data')

    scenarios = ["zero", "party", "diverge", "estimate"]
    seed = FLAGS.seed

    for s in range(97, 115):
        print('Starting session ' + str(s) + '.')
        ## Trigger different scenarios
        for scenario in scenarios:
            print('Scenario = ' + scenario)
            ## Directory setup
            s_dir = os.path.join(source_dir, FLAGS.simulation + '-' + scenario + '-' + str(s))
            input_dir = os.path.join(s_dir, 'clean')
            fig_dir = os.path.join(s_dir, 'fig')
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)

            ## Load data
            author_indices = np.load(os.path.join(input_dir, "author_indices.npy")).astype(np.int32)
            author_data = np.loadtxt(os.path.join(input_dir, "author_map.txt"),
                                     dtype=str, delimiter=" ", usecols=[0, 1, -1])
            author_party = np.char.replace(author_data[:, 2], '(', '')
            author_party = np.char.replace(author_party, ')', '')
            author_map = np.char.add(author_data[:, 0], author_data[:, 1])
            document_party = tf.gather(author_party, author_indices)
            doc_D_indices = tf.where(document_party == 'D')[:, 0]
            doc_R_indices = tf.where(document_party == 'R')[:, 0]

            counts = sparse.load_npz(os.path.join(input_dir, "counts" + scenario + ".npz"))
            print(counts.shape)
            # aggregate over party
            counts_D = counts[doc_D_indices, ]
            sums_D = tf.gather(counts_D.sum(axis=0), 0, axis=0)
            counts_R = counts[doc_R_indices, ]
            sums_R = tf.gather(counts_R.sum(axis=0), 0, axis=0)
            sums = sums_D + sums_R
            totsum_D = sum(sums_D)
            totsum_R = sum(sums_R)
            # print the differences
            # contributions to chi_square statistics by party
            null_D = sums_D * totsum_D / (totsum_D + totsum_R)
            chi_D = tf.square(sums_D - null_D) / null_D
            null_R = sums_R * totsum_R / (totsum_D + totsum_R)
            chi_R = tf.square(sums_R - null_R) / null_R
            chi = sum(chi_D) + sum(chi_R)
            dif = sums_D - sums_R
            print(dif)
            # print(max(dif))
            # print(min(dif))
            fig, axs = plt.subplots(1, 3,
                                    #sharey=True,
                                    figsize=(9, 9), tight_layout=True)
            axs[0].hist(tf.gather(dif, tf.where(tf.abs(dif) < 25)[:, 0]))
            axs[1].hist(tf.gather(chi_D, tf.where(tf.abs(chi_D) < 25)[:, 0]))
            axs[2].hist(tf.gather(chi_R, tf.where(tf.abs(chi_R) < 25)[:, 0]))
            plt.savefig(os.path.join(fig_dir, 'hist_sum_counts_D_minus_R.pdf'),
                        bbox_inches='tight')  # uncomment to save
            # plt.show()
            plt.close()

        print('Session ' + str(s) + ' finished')

if __name__ == '__main__':
    app.run(main)
