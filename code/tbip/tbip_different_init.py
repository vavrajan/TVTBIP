"""Learn ideal points with the text-based ideal point model (TBIP).

Let y_{dv} denote the counts of word v in document d. Let x_d refer to the
ideal point of the author of document d. Then we model:

theta, beta ~ Gamma(alpha, alpha)
x, eta ~ N(0, 1)
y_{dv} ~ Pois(sum_k theta_dk beta_kv exp(x_d * eta_kv).

We perform variational inference to provide estimates for the posterior
distribution of each latent variable. We take reparameterization gradients,
using a lognormal variational family for the positive variables (theta, beta)
and a normal variational family for the real variables (x, eta).

The directory `data/{data_name}/clean/` should have the following four files:
  1. `counts.npz`: a [num_documents, num_words] sparse matrix containing the
     word counts for each document.
  2. `author_indices.npy`: a [num_documents] vector where each entry is an
     integer in the set {0, 1, ..., num_authors - 1}, indicating the author of
     the corresponding document in `counts.npz`.
  3. `vocabulary.txt`: a [num_words]-length file where each line is a string
     denoting the corresponding word in the vocabulary.
  4. `author_map.txt`: a [num_authors]-length file where each line is a string
     denoting the name of an author in the corpus.

We provide more details in our paper [1].

#### References
[1]: Keyon Vafa, Suresh Naidu, David Blei. Text-Based Ideal Points. In
     _Conference of the Association for Computational Linguistics_, 2020.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import flags
import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.decomposition import non_negative_factorization, NMF

# import pandas as pd


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_float("epsilon",
                   default=1e-7,
                   help="Epsilon for Adam optimizer.")
flags.DEFINE_integer("max_steps",
                     default=1000000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_topics",
                     default=50,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=1024,
                     help="Batch size.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "binary", "sqrt", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_enum("pre_initialize_parameters",
                  default='PF',
                  enum_values=['None', 'NMF', 'PF', 'previous'],
                  help="How to initialize thetas (document), betas (neutral topics) and etas (ideological topics):"
                       "None = random initialization,"
                       "NMF = initialize theta and beta with NMF, estimates + 1e-3,"
                       "PF = initialize with Poisson Factorization (has to be run before, loc = shapes / rate),"
                       "previous =  1. take estimated betas from previous data with the same vocabulary,"
                       "            2. estimate thetas by non_negative_factorization with fixed H=beta,"
                       "            3. use log(thetas + 1e-3), log(betas) and eta locations to "
                       "                initialize variational families.")
flags.DEFINE_string("data",
                    default="hein-daily-114",
                    help="Data source being used.")
flags.DEFINE_string("previous_data",
                    default="hein-daily-113",
                    help="Previous data source used for initialization, if pre_initialize_parameters=previous.")
flags.DEFINE_integer("print_steps",
                     default=500,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def build_input_pipeline(data_dir,
                         batch_size,
                         random_state,
                         counts_transformation="nothing"):
    """Load data and build iterator for minibatches.

    Args:
      data_dir: The directory where the data is located. There must be four
        files inside the rep: `counts.npz`, `author_indices.npy`,
        `author_map.txt`, and `vocabulary.txt`.
      batch_size: The batch size to use for training.
      random_state: A NumPy `RandomState` object, used to shuffle the data.
      counts_transformation: A string indicating how to transform the counts.
        One of "nothing", "binary", "log", or "sqrt".
    """
    counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
    num_documents, num_words = counts.shape
    author_indices = np.load(
        os.path.join(data_dir, "author_indices.npy")).astype(np.int32)
    num_authors = np.max(author_indices + 1)
    author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                            dtype=str,
                            delimiter="\n")
    # Shuffle data.
    documents = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[documents]
    shuffled_counts = counts[documents]

    # Apply counts transformation.
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "binary":
        count_values = np.int32(shuffled_counts.data > 0)
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    elif counts_transformation == "sqrt":
        count_values = np.round(np.sqrt(shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")

    # Store counts as sparse tensor so it occupies less memory.
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (documents, shuffled_counts, shuffled_author_indices))
    batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
    iterator = batches.make_one_shot_iterator()
    vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                            dtype=str,
                            delimiter="\n",
                            comments="<!-")

    total_counts_per_author = np.bincount(
        author_indices,
        weights=np.array(np.sum(counts, axis=1)).flatten())
    counts_per_document_per_author = (
            total_counts_per_author / np.bincount(author_indices))
    # Author weights is how much lengthy each author's opinion over average is.
    author_weights = (counts_per_document_per_author /
                      np.mean(np.sum(counts, axis=1))).astype(np.float32)
    return (iterator, author_weights, vocabulary, author_map,
            num_documents, num_words, num_authors)


def build_lognormal_variational_parameters(initial_document_loc,
                                           initial_objective_topic_loc,
                                           num_documents,
                                           num_words,
                                           num_topics):
    """
    Build document and objective topic lognormal variational parameters.

    Args:
      initial_document_loc: A [num_documents, num_topics] NumPy array containing
        the initial document intensity means.
      initial_objective_topic_loc: A [num_topics, num_words] NumPy array
        containing the initial objective topic means.
      num_documents: Number of documents in the data set.
      num_words: Number of words in the data set.
      num_topics: Number of topics.

    Returns:
      document_loc: A Variable object with shape [num_documents, num_topics].
      document_scale: A positive Variable object with shape [num_documents,
        num_topics].
      objective_topic_loc: A Variable object with shape [num_topics, num_words].
      objective_topic_scale: A positive Variable object with shape [num_topics,
        num_words].
    """
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(np.log(initial_objective_topic_loc)))
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, num_topics],
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        dtype=tf.float32)
    objective_topic_scale_logit = tf.get_variable(
        "objective_topic_scale_logit",
        shape=[num_topics, num_words],
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)

    tf.summary.histogram("params/document_loc", document_loc)
    tf.summary.histogram("params/objective_topic_loc", objective_topic_loc)
    tf.summary.histogram("params/document_scale", document_scale)
    tf.summary.histogram("params/objective_topic_scale", objective_topic_scale)

    return (document_loc, document_scale,
            objective_topic_loc, objective_topic_scale)


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
    """Get neutral and ideological topics to be used for Tensorboard.

    Args:
      neutral_mean: The mean of the neutral topics, a NumPy matrix with shape
        [num_topics, num_words].
      negative_mean: The mean of the negative topics, a NumPy matrix with shape
        [num_topics, num_words].
      positive_mean: The mean of the positive topics, a NumPy matrix with shape
        [num_topics, num_words].
      vocabulary: A list of the vocabulary with shape [num_words].

    Returns:
      topic_strings: A list of the negative, neutral, and positive topics.
    """
    num_topics, num_words = neutral_mean.shape
    words_per_topic = 10
    top_neutral_words = np.argsort(-neutral_mean, axis=1)
    top_negative_words = np.argsort(-negative_mean, axis=1)
    top_positive_words = np.argsort(-positive_mean, axis=1)
    topic_strings = []
    for topic_idx in range(num_topics):
        neutral_start_string = "Neutral {}:".format(topic_idx)
        neutral_row = [vocabulary[word] for word in
                       top_neutral_words[topic_idx, :words_per_topic]]
        neutral_row_string = ", ".join(neutral_row)
        neutral_string = " ".join([neutral_start_string, neutral_row_string])

        positive_start_string = "Positive {}:".format(topic_idx)
        positive_row = [vocabulary[word] for word in
                        top_positive_words[topic_idx, :words_per_topic]]
        positive_row_string = ", ".join(positive_row)
        positive_string = " ".join([positive_start_string, positive_row_string])

        negative_start_string = "Negative {}:".format(topic_idx)
        negative_row = [vocabulary[word] for word in
                        top_negative_words[topic_idx, :words_per_topic]]
        negative_row_string = ", ".join(negative_row)
        negative_string = " ".join([negative_start_string, negative_row_string])

        topic_strings.append("  \n".join(
            [negative_string, neutral_string, positive_string]))
    return np.array(topic_strings)


def print_ideal_points(ideal_point_loc, author_map):
    """Print ideal point ordering for Tensorboard."""
    return ", ".join(author_map[np.argsort(ideal_point_loc)])


def get_log_prior(samples, prior):
    """Return log prior of sampled Gaussians.

    Args:
      samples: A `Tensor` with shape `[num_samples, :, :]`.
      prior: String representing prior distribution.

    Returns:
      log_prior: A `Tensor` with shape `[num_samples]`, with the log priors
        summed across latent dimensions.
    """
    if prior == 'normal':
        prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
    elif prior == 'gamma':
        prior_distribution = tfp.distributions.Gamma(concentration=0.3, rate=0.3)
    log_prior = tf.reduce_sum(prior_distribution.log_prob(samples),
                              axis=[1, 2])
    return log_prior


def get_elbo(counts,
             document_indices,
             author_indices,
             author_weights,
             document_distribution,
             objective_topic_distribution,
             ideological_topic_distribution,
             ideal_point_distribution,
             num_documents,
             batch_size,
             num_samples=1):
    """Approximate variational Lognormal ELBO using reparameterization.

    Args:
      counts: A matrix with shape `[batch_size, num_words]`.
      document_indices: An int-vector with shape `[batch_size]`.
      author_indices: An int-vector with shape `[batch_size]`.
      author_weights: A vector with shape `[num_authors]`, constituting how
        lengthy the opinion is above average.
      document_distribution: A positive `Distribution` object with parameter
        shape `[num_documents, num_topics]`.
      objective_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideological_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideal_point_distribution: A `Distribution` object over [0, 1] with
        parameter_shape `[num_authors]`.
      num_documents: The number of documents in the total data set (used to
        calculate log-likelihood scale).
      batch_size: Batch size (used to calculate log-likelihood scale).
      num_samples: Number of Monte-Carlo samples.

    Returns:
      elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
        averaged across samples and summed across batches.
    """
    document_samples = document_distribution.sample(num_samples)
    objective_topic_samples = objective_topic_distribution.sample(num_samples)
    ideological_topic_samples = ideological_topic_distribution.sample(
        num_samples)
    ideal_point_samples = ideal_point_distribution.sample(num_samples)

    _, num_topics, _ = objective_topic_samples.get_shape().as_list()

    ideal_point_log_prior = tfp.distributions.Normal(
        loc=0.,
        scale=1.)
    ideal_point_log_prior = tf.reduce_sum(
        ideal_point_log_prior.log_prob(ideal_point_samples), axis=1)

    document_log_prior = get_log_prior(document_samples, 'gamma')
    objective_topic_log_prior = get_log_prior(objective_topic_samples, 'gamma')
    ideological_topic_log_prior = get_log_prior(ideological_topic_samples,
                                                'normal')
    log_prior = (document_log_prior +
                 objective_topic_log_prior +
                 ideological_topic_log_prior +
                 ideal_point_log_prior)

    selected_document_samples = tf.gather(document_samples,
                                          document_indices,
                                          axis=1)
    selected_ideal_points = tf.gather(ideal_point_samples,
                                      author_indices,
                                      axis=1)

    selected_ideological_topic_samples = tf.exp(
        selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *
        ideological_topic_samples[:, tf.newaxis, :, :])
    # Normalize by how lengthy the author's opinion is.
    selected_author_weights = tf.gather(author_weights, author_indices)
    selected_ideological_topic_samples = (
            selected_author_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
            selected_ideological_topic_samples)

    document_entropy = -tf.reduce_sum(
        document_distribution.log_prob(document_samples),
        axis=[1, 2])
    objective_topic_entropy = -tf.reduce_sum(
        objective_topic_distribution.log_prob(objective_topic_samples),
        axis=[1, 2])
    ideological_topic_entropy = -tf.reduce_sum(
        ideological_topic_distribution.log_prob(ideological_topic_samples),
        axis=[1, 2])
    ideal_point_entropy = -tf.reduce_sum(
        ideal_point_distribution.log_prob(ideal_point_samples),
        axis=1)
    entropy = (document_entropy +
               objective_topic_entropy +
               ideological_topic_entropy +
               ideal_point_entropy)

    rate = tf.reduce_sum(
        selected_document_samples[:, :, :, tf.newaxis] *
        objective_topic_samples[:, tf.newaxis, :, :] *
        selected_ideological_topic_samples[:, :, :, :],
        axis=2)

    count_distribution = tfp.distributions.Poisson(rate=rate)
    # Need to un-sparsify the counts to evaluate log-likelihood.
    count_log_likelihood = count_distribution.log_prob(
        tf.sparse.to_dense(counts))
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    count_log_likelihood = count_log_likelihood * (num_documents / batch_size)

    elbo = log_prior + count_log_likelihood + entropy
    elbo = tf.reduce_mean(elbo)

    tf.summary.scalar("elbo/elbo", elbo)
    tf.summary.scalar("elbo/log_prior", tf.reduce_mean(log_prior))
    tf.summary.scalar("elbo/count_log_likelihood",
                      tf.reduce_mean(count_log_likelihood))
    tf.summary.scalar("elbo/entropy", tf.reduce_mean(entropy))
    return elbo


def main(argv):
    del argv
    tf.set_random_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)

    project_dir = os.path.abspath(os.path.dirname(__file__))
    source_dir = os.path.join(project_dir, "data", FLAGS.data)
    prev_dir = os.path.join(project_dir, "data", FLAGS.previous_data)
    # As described in the docstring, the data directory must have the following
    # files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
    data_dir = os.path.join(source_dir, "clean")
    save_dir = os.path.join(source_dir, "tbip-fits")
    prev_dir = os.path.join(prev_dir, "tbip-fits", "params")
    if tf.gfile.Exists(save_dir):
        tf.logging.warn("Deleting old log directory at {}".format(save_dir))
        tf.gfile.DeleteRecursively(save_dir)
    tf.gfile.MakeDirs(save_dir)

    (iterator, author_weights, vocabulary, author_map,
     num_documents, num_words, num_authors) = build_input_pipeline(
        data_dir,
        FLAGS.batch_size,
        random_state,
        FLAGS.counts_transformation)

    document_indices, counts, author_indices = iterator.get_next()

    if FLAGS.pre_initialize_parameters == 'NMF':
        X = sparse.load_npz(os.path.join(data_dir, 'counts.npz'))
        nmf_model = NMF(n_components=FLAGS.num_topics,
                        init='random',
                        random_state=0,
                        max_iter=500)
        # Add offset to make sure none are zero.
        initial_document_loc = np.float32(nmf_model.fit_transform(X) + 1e-3)
        initial_objective_topic_loc = np.float32(nmf_model.components_ + 1e-3)
        initial_ideological_topic_loc = None
        initial_ideological_topic_scale_logit_prev = None
        initial_ideological_topic_loc_shape = [FLAGS.num_topics, num_words]
    elif FLAGS.pre_initialize_parameters == 'PF':
        fit_dir = os.path.join(source_dir, "pf-fits")
        fitted_document_shape = np.load(os.path.join(fit_dir, "document_shape.npy")).astype(np.float32)
        fitted_document_rate = np.load(os.path.join(fit_dir, "document_rate.npy")).astype(np.float32)
        fitted_topic_shape = np.load(os.path.join(fit_dir, "topic_shape.npy")).astype(np.float32)
        fitted_topic_rate = np.load(os.path.join(fit_dir, "topic_rate.npy")).astype(np.float32)
        initial_document_loc = fitted_document_shape / fitted_document_rate
        initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate
        initial_ideological_topic_loc = None
        initial_ideological_topic_scale_logit_prev = None
        initial_ideological_topic_loc_shape = [FLAGS.num_topics, num_words]
    elif FLAGS.pre_initialize_parameters == 'previous':
        objective_topic_loc = np.load(os.path.join(prev_dir, "objective_topic_loc.npy"))
        beta = np.exp(objective_topic_loc)
        X = sparse.load_npz(os.path.join(data_dir, 'counts.npz'))
        # X = X.astype('float64')
        W, H, n_iter = non_negative_factorization(X, n_components=FLAGS.num_topics,
                                                  init='custom',
                                                  random_state=0,
                                                  update_H=False,
                                                  H=beta.astype('float32'))
        W = W.astype('float32')
        initial_document_loc = W + 1e-3 # to have non-zero theta
        initial_objective_topic_loc = beta
        ideological_topic_loc_prev = np.load(os.path.join(prev_dir, 'ideological_topic_loc.npy'))  # eta
        ideological_topic_loc_prev[ideological_topic_loc_prev < -1] = -1.0
        ideological_topic_loc_prev[ideological_topic_loc_prev > 1] = 1.0
        ideological_topic_scl_prev = np.load(os.path.join(prev_dir, 'ideological_topic_scale.npy'))  # eta scale
        initial_ideological_topic_loc = tf.convert_to_tensor(np.nan_to_num(ideological_topic_loc_prev, nan=0.0))
        initial_ideological_topic_scale_logit_prev = tfp.math.softplus_inverse(
            tf.convert_to_tensor(ideological_topic_scl_prev))
        initial_ideological_topic_loc_shape = None
    else:
        initial_document_loc = np.float32(
            np.exp(random_state.randn(num_documents, FLAGS.num_topics)))
        initial_objective_topic_loc = np.float32(
            np.exp(random_state.randn(FLAGS.num_topics, num_words)))
        initial_ideological_topic_loc = None
        initial_ideological_topic_scale_logit_prev = None
        initial_ideological_topic_loc_shape = [FLAGS.num_topics, num_words]

    # Initialize lognormal variational parameters.
    (document_loc, document_scale, objective_topic_loc,
     objective_topic_scale) = build_lognormal_variational_parameters(
        initial_document_loc,
        initial_objective_topic_loc,
        num_documents,
        num_words,
        FLAGS.num_topics)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)
    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)

    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        initializer=initial_ideological_topic_loc,
        shape=initial_ideological_topic_loc_shape,
        dtype=tf.float32)
    ideological_topic_scale_logit = tf.get_variable(
        "ideological_topic_scale_logit",
        # initializer=initial_ideological_topic_scale_logit_prev,
        shape=[FLAGS.num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
    tf.summary.histogram("params/ideological_topic_loc", ideological_topic_loc)
    tf.summary.histogram("params/ideological_topic_scale", ideological_topic_scale)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)

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
    tf.summary.histogram("params/ideal_point_loc",
                         tf.reshape(ideal_point_loc, [-1]))
    tf.summary.histogram("params/ideal_point_scale",
                         tf.reshape(ideal_point_scale, [-1]))

    elbo = get_elbo(counts,
                    document_indices,
                    author_indices,
                    author_weights,
                    document_distribution,
                    objective_topic_distribution,
                    ideological_topic_distribution,
                    ideal_point_distribution,
                    num_documents,
                    FLAGS.batch_size,
                    num_samples=FLAGS.num_samples)
    loss = -elbo
    tf.summary.scalar("loss", loss)

    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    train_op = optim.minimize(loss)

    """
    For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)], 
    and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to the 
    variational distributions. Like the paper, beta refers to the obective topic
    and eta refers to the ideological topic.

    Dropping the indices and denoting by mu_b the objective topic location and 
    sigma_b the objective topic scale, we have E[beta] = exp(mu + sigma_b^2 / 2),
    using the mean of a lognormal distribution.

    Denoting by mu_e the ideological topic location and sigma_e the ideological
    topic scale, we have E[beta * exp(eta)] = E[beta]E[exp(eta)] by the 
    mean-field assumption. exp(eta) is lognormal distributed, so E[exp(eta)] =
    exp(mu_e + sigma_e^2 / 2). Thus, E[beta * exp(eta)] = 
    exp(mu_b + mu_e + (sigma_b^2 + sigma_e^2) / 2).

    Finally, E[beta * exp(-eta)] = 
    exp(mu_b - mu_e + (sigma_b^2 + sigma_e^2) / 2).

    Because we only care about the orderings of topics, we can drop the exponents
    from the means.
    """
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

    topics = tf.py_func(
        functools.partial(print_topics, vocabulary=vocabulary),
        [neutral_mean, negative_mean, positive_mean],
        tf.string,
        stateful=False)
    ideal_point_list = tf.py_func(
        functools.partial(print_ideal_points, author_map=author_map),
        [ideal_point_loc],
        tf.string, stateful=False)
    tf.summary.text("topics", topics)
    tf.summary.text("ideal_points", ideal_point_list)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
        sess.run(init)
        start_time = time.time()
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str)
        summary_writer.flush()
        for step in range(FLAGS.max_steps):
            (_, elbo_val) = sess.run([train_op, elbo])
            duration = (time.time() - start_time) / (step + 1)
            if step % FLAGS.print_steps == 0:
                print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
                    step, elbo_val, duration))

                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 1000 == 0 or step == FLAGS.max_steps - 1:
                param_save_dir = os.path.join(save_dir, "params")
                if not tf.gfile.Exists(param_save_dir):
                    tf.gfile.MakeDirs(param_save_dir)

                # (ideological_topic_loc_val, ideological_topic_scale_val,
                #  ideal_point_loc_val, ideal_point_scale_val) = sess.run([
                #     ideological_topic_loc, ideological_topic_scale,
                #     ideal_point_loc, ideal_point_scale])

                (document_loc_val, document_scale_val, objective_topic_loc_val,
                 objective_topic_scale_val, ideological_topic_loc_val,
                 ideological_topic_scale_val, ideal_point_loc_val,
                 ideal_point_scale_val,
                 doc_mean, neu_mean, neg_mean, pos_mean) = sess.run([
                    document_loc, document_scale, objective_topic_loc,
                    objective_topic_scale, ideological_topic_loc,
                    ideological_topic_scale, ideal_point_loc, ideal_point_scale,
                    document_mean, neutral_mean, negative_mean, positive_mean])
                np.save(os.path.join(param_save_dir, "document_loc"),
                        document_loc_val)
                np.save(os.path.join(param_save_dir, "document_scale"),
                        document_scale_val)
                np.save(os.path.join(param_save_dir, "objective_topic_loc"),
                        objective_topic_loc_val)
                np.save(os.path.join(param_save_dir, "objective_topic_scale"),
                        objective_topic_scale_val)
                np.save(os.path.join(param_save_dir, "ideological_topic_loc"),
                        ideological_topic_loc_val)
                np.save(os.path.join(param_save_dir, "ideological_topic_scale"),
                        ideological_topic_scale_val)
                np.save(os.path.join(param_save_dir, "ideal_point_loc"),
                        ideal_point_loc_val)
                np.save(os.path.join(param_save_dir, "ideal_point_scale"),
                        ideal_point_scale_val)
                np.save(os.path.join(param_save_dir, "document_mean"), doc_mean)
                np.save(os.path.join(param_save_dir, "neutral_topic_mean"), neu_mean)
                np.save(os.path.join(param_save_dir, "negative_topic_mean"), neg_mean)
                np.save(os.path.join(param_save_dir, "positive_topic_mean"), pos_mean)

                # speaker_IP = pd.DataFrame(columns=['speaker', 'ideal_point'])  # create an empty dataframe
                # speaker_IP['speaker'] = author_map
                # speaker_IP['ideal_point'] = ideological_topic_loc_val
                # speaker_IP.to_csv(os.path.join(param_save_dir, "ideal_point_speakers.csv"), header=True)


if __name__ == "__main__":
    tf.app.run()
