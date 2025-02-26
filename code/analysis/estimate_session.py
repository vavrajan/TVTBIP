
import functools
import time
import os
import csv
import pandas as pd

# import tensorflow_version 1.15
import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import NMF, non_negative_factorization
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app
from absl import flags

import sys

## FLAGS
flags.DEFINE_enum("scenario", default="", enum_values=["", "_zero", "_party", "_diverge", "_estimate"],
                  help="Simulation scenario, one of:"
                       "= use the original counts,"
                       "_zero = all ideological positions will be zero,"
                       "_party = all ideological positions will be determined by party -1, 0, 1,"
                       "_diverge = ideal will start at zero and from session 10* grows away from zero,"
                       "_estimate = use estimates of ideal from the corresponding sessions.")
flags.DEFINE_integer("seed2", default=0, help="Random seed to be used.")
flags.DEFINE_string("data_name", default='hein-daily', help="Name of the dataset.")
flags.DEFINE_integer("session", default=97, help="Session number (time-point) to estimate.")
flags.DEFINE_float("eps", default=1, help="Epsilon (precision) of the Adam algorithm.")
flags.DEFINE_float("learningrate", default=0.0001, help="Learning rate of the Adam algorithm.")
flags.DEFINE_integer("numtopics", default=25, help="Number of topics.")
flags.DEFINE_integer("batchsize", default=512, help="Number of documents in a single batch.")
flags.DEFINE_integer("maxsteps", default=250000, help="Maximum number of iterations.")
flags.DEFINE_integer("printsteps", default=25000, help="How often to print the steps.")
flags.DEFINE_enum("initialization", default="NMF", enum_values=["NMF", "random", "previous"],
                  help="The way model parameters should be initialized:"
                       "NMF = theta and beta by non-negative matrix factorization (use only for the first time point!),"
                       "random = theta and beta by random values (use only for the first time point!),"
                       "previous = beta and eta estimates from the previously estimated session and fit of non_negative_factorization with known beta for parameter theta.")
flags.DEFINE_boolean("initialize_ideal_by_party", default=False, help="Initialize the ideological positions by political party? D=1, I=0, R=-1?")
FLAGS = flags.FLAGS

def main(argv):
    del argv
    print(tf.__version__)

    tf.test.gpu_device_name()
    tf.set_random_seed(FLAGS.seed2)
    random_state = np.random.RandomState(FLAGS.seed2)

    num_topics = FLAGS.numtopics
    initialization = FLAGS.initialization
    #parameters for ADAM
    eps = FLAGS.eps
    learning_rate = FLAGS.learningrate
    batch_size = FLAGS.batchsize
    max_steps = FLAGS.maxsteps
    print_steps = FLAGS.printsteps


    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name)
    s_dir = os.path.join(source_dir, str(FLAGS.session))
    prev_dir = os.path.join(source_dir, str(FLAGS.session - 1))
    input_dir = os.path.join(s_dir, 'input')
    output_dir = os.path.join(s_dir, 'output')
    prev_input_dir = os.path.join(prev_dir, 'input')
    prev_output_dir = os.path.join(prev_dir, 'output')
    #path to tbip.py, to change as needed
    # py_file = os.path.join(project_dir, 'jae_revision')
    py_file = os.path.join(project_dir, 'python', FLAGS.data_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sys.path.append(os.path.abspath(py_file))
    import tbip

    scenario = FLAGS.scenario

    counts_path = os.path.join(input_dir, 'counts' + scenario + '.npz')
    counts = sparse.load_npz(counts_path)
    num_documents, num_words = counts.shape

    author_data = np.loadtxt(os.path.join(input_dir, "author_map.txt"),
                             dtype=str, delimiter=" ", usecols=[0, 1, -1])
    author_party = np.char.replace(author_data[:, 2], '(', '')
    author_party = np.char.replace(author_party, ')', '')

    init_ideal_loc = 1.0*(author_party == 'D') + 0.0*(author_party == 'I') - 1.0*(author_party == 'R')

    if initialization == "NMF":
        nmf_model = NMF(n_components=num_topics,
                        init='random',
                        random_state=0,
                        max_iter=500)
        # Add offset to make sure none are zero.
        initial_document_loc = np.float32(nmf_model.fit_transform(counts) + 1e-3)
        initial_objective_topic_loc = np.float32(nmf_model.components_ + 1e-3)
    elif initialization == "random":
        initial_document_loc = np.float32(
           np.exp(random_state.randn(num_documents, num_topics)))
        initial_objective_topic_loc = np.float32(
           np.exp(random_state.randn(num_topics, num_words)))
    elif initialization == "previous":
        objective_topic_loc = np.load(os.path.join(prev_output_dir, "objective_topic_loc" + scenario + ".npy"))
        beta = np.exp(objective_topic_loc)

        W, H, n_iter = non_negative_factorization(counts,
                                                  n_components=num_topics,
                                                  init='custom',
                                                  random_state=0,
                                                  update_H=False,
                                                  H=beta.astype('float32'))

        W = W.astype('float32')
        W = W + 1e-3  # causes 0 value otherwise
        initial_document_loc = W  # theta
        initial_objective_topic_loc = beta.astype('float32')  # don't need, use objective_topic_loc

        ideological_topic_loc_prev = np.load(os.path.join(prev_output_dir,
                                                          'ideological_topic_loc' + scenario + '.npy'))  # eta
        initial_ideological_topic_loc = tf.convert_to_tensor(ideological_topic_loc_prev)
    else:
        raise ValueError("Unknown choice for initilization of model parameters, choose one of NMF, random or previous.")

    (iterator, author_weights, vocabulary, author_map,
     num_documents, num_words, num_authors) = tbip.build_input_pipeline(
        counts_path,
        input_dir,
        batch_size,
        random_state,
        counts_transformation='nothing')
    document_indices, counts, author_indices = iterator.get_next()

    # Create Lognormal variational family for document intensities (theta).
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, num_topics],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)

    # Create Lognormal variational family for objective topics (beta).
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(np.log(initial_objective_topic_loc)))
    objective_topic_scale_logit = tf.get_variable(
        "objective_topic_scale_logit",
        shape=[num_topics, num_words],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32)
    objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)
    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)

    # Create Gaussian variational family for ideological topics (eta).
    if initialization == "previous":
        ideological_topic_loc = tf.get_variable(
            "ideological_topic_loc",
            initializer=initial_ideological_topic_loc,
            trainable=True)
    else:
        ideological_topic_loc = tf.get_variable(
            "ideological_topic_loc",
            shape=[num_topics, num_words],
            dtype=tf.float32)
    ideological_topic_scale_logit = tf.get_variable(
        "ideological_topic_scale_logit",
        shape=[num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)

    # Create Gaussian variational family for ideal points (x).
    if FLAGS.initialize_ideal_by_party:
        ideal_point_loc = tf.get_variable(
            "ideal_point_loc",
            initializer=tf.constant(np.float32(init_ideal_loc)),
            dtype=tf.float32)
    else:
        ideal_point_loc = tf.get_variable(
            "ideal_point_loc",
            shape=[num_authors],
            dtype=tf.float32)
    ideal_point_scale_logit = tf.get_variable(
        "ideal_point_scale_logit",
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
    ideal_point_distribution = tfp.distributions.Normal(
        loc=ideal_point_loc,
        scale=ideal_point_scale)

    # Approximate ELBO.
    elbo = tbip.get_elbo(counts,
                         document_indices,
                         author_indices,
                         author_weights,
                         document_distribution,
                         objective_topic_distribution,
                         ideological_topic_distribution,
                         ideal_point_distribution,
                         num_documents,
                         batch_size)
    loss = -elbo

    optim = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=eps)
    train_op = optim.minimize(loss)

    document_mean = document_loc + document_scale ** 2 / 2

    neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2

    positive_mean = (objective_topic_loc +
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)

    negative_mean = (objective_topic_loc -
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)

    print('Learning for scenario ' + scenario)

    loss_vals = []
    ideal_df = pd.DataFrame(data={"speaker": author_map, "party": author_party})
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    start_time = time.time()

    for step in range(max_steps):
        (_, elbo_val) = sess.run([train_op, elbo])
        duration = (time.time() - start_time) / (step + 1)
        loss_vals.append(elbo_val) #Keeping a track of elbo values
        if step % print_steps == 0 or step == max_steps - 1:
            print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec/step)".format(
            step, elbo_val, duration))
        if (step + 1) % print_steps == 0 or step == max_steps - 1:
            (document_topic_mean, neutral_topic_mean, negative_topic_mean, positive_topic_mean,
             ideal_point_mean, ots, otl, itl, its) = sess.run([document_mean, neutral_mean, negative_mean,
                                           positive_mean, ideal_point_loc, objective_topic_scale, objective_topic_loc,
                                           ideological_topic_loc, ideological_topic_scale, ])
            ideal_df['ideal' + str(step)] = ideal_point_mean
            ideal_df.to_csv(os.path.join(output_dir, 'ideal_step_by_step'+scenario+'.csv'), sep=',', index=False)
            np.save(os.path.join(output_dir, "document_topic_mean" + scenario + ".npy"), document_topic_mean)
            np.save(os.path.join(output_dir, "neutral_topic_mean" + scenario + ".npy"), neutral_topic_mean)
            np.save(os.path.join(output_dir, "negative_topic_mean" + scenario + ".npy"), negative_topic_mean)
            np.save(os.path.join(output_dir, "positive_topic_mean" + scenario + ".npy"), positive_topic_mean)
            np.save(os.path.join(output_dir, "ideal_point_mean" + scenario + ".npy"), ideal_point_mean)
            np.save(os.path.join(output_dir, "objective_topic_scale" + scenario + ".npy"), ots)
            np.save(os.path.join(output_dir, "objective_topic_loc" + scenario + ".npy"), otl)
            np.save(os.path.join(output_dir, "ideological_topic_loc" + scenario + ".npy"), itl)
            np.save(os.path.join(output_dir, "ideological_topic_scale" + scenario + ".npy"), its)
        loss_df = pd.DataFrame(data={"loss": loss_vals})
        loss_df.to_csv(os.path.join(output_dir, 'loss_values' + scenario + '.csv'), sep=',', index=False) #save loss values as a csv file


if __name__ == '__main__':
    app.run(main)