# Time Varying Text-Based Ideal Point Model
Source code for the paper: 
[Revisiting Group Differences in High-Dimensional Choices: Methods and Application to Congressional Speech by Paul Hofmarcher, Jan Vávra, Sourav Adhikari, and Bettina Grün (2025)](link).

This repository contains the code for
* preprocessing the [Hein-Daily](https://data.stanford.edu/congress_text) dataset of congressional speeches,
* estimating TBIP provided by [Keyon Vafa](https://github.com/keyonvafa/tbip),
* estimating our Time-Varying version of TBIP,
* simulation study that imitates the original dataset.

### Data preparation

In order to perform the analysis of the congressional speeches, please:
* download the data `hein-daily.zip` from [Hein-Daily](https://data.stanford.edu/congress_text),
* unzip them into the directory [data/hein-daily](data/hein-daily),
* add your own file with [stopwords](data/hein-daily/stopwords.txt) (ours copied from [Keyon Vafa](https://github.com/keyonvafa/tbip)),
* run the two scripts in `code/preprocessing`: 
  * [01_supervocab.py](code/preprocessing/01_supervocab.py) to create vocabulary that spans all sessions,
  * [02_inputmatrices.py](code/preprocessing/02_inputmatrices.py) to create the count matrices and other required files.

If you wish to use TVTBIP for your own dataset, you need to keep the following structure:
* `data/your-data-name` should be the directory with the original data format
* `data/your-data-name-number` should store both clean data and the estimates from TVTBIP for time period with this number
  * for `hein-daily` we have [97](data/hein-daily-97), ..., [114](data/hein-daily-114)
* within the subdirectory for a time period should be two subdirectories to store:
  * [clean](data/hein-daily-97/clean) the clean inputs for the TVTBIP, which should contain:
    * `counts.npz` - the document-term matrix in sparse format,
    * `author_indices.npy` - vector of author indices declaring who is the author of the corresponding document,
    * `author_map.txt` - name and surname of the speaker followed by party label in parentheses (one per line),
    * `vocabulary.txt` - list of words (one per line) that should be common to all sessions for TVTBIP,
  * [pf-fits](data/hein-daily-97/pf-fits) the initial parameter values estimated with Poisson factorization for the first time period,
  * [tbip-fits](data/hein-daily-97/tbip-fits) outputs of the TVTBIP, which will contain subdirectory `param` with estimated parameter values.

### Estimating TVTBIP

The estimation is performed by [tbip_different_init.py](code/tbip/tbip_different_init.py), 
which runs only one single session (time period). 
It is the original [tbip.py](code/tbip/tbip.py) model by [Keyon Vafa](https://github.com/keyonvafa/tbip)
but the initialization process is adjusted to reflect our 
time-varying version. 
Both were designed for Tensorflow version 1.15 which 
substantially differs from the newer version.
The list of the versions of the key libraries is below:
* absl-py                1.4.0 
* matplotlib             3.3.4 
* numpy                  1.19.5 
* pandas                 1.1.5
* pip                    21.3.1
* scikit-learn           0.24.2
* scipy                  1.5.2
* seaborn                0.11.2
* tensorboard            1.15.0
* tensorflow             1.15.0
* tensorflow-estimator   1.15.1
* tensorflow-gpu         1.15.0
* tensorflow-probability 0.8.0rc0
* wordcloud              1.9.2

More details could be found in [requirements_tf_1_15](requirements_tf_1_15.txt).

Depending on the flag `pre_initialize_parameters` you have four options:
* `random` - initialize model parameters completely at random, which we do not recommend,
* `NMF` - initialize the location parameters for documents and objective topics by performing Non-negative Matrix Factorisation (NMF) first, 
we use `NMF` from `sklearn.decomposition`:
  * used for the first time period (session),
* `PF` - initialize the location parameters for documents and objective topics by performing Poisson Factorization (PF) first, 
where the implementation from [tbip](https://github.com/keyonvafa/tbip) is used:
  * alternative initialization of the first time period (session),
* `previous` - for time period `t` initialize model parameters by estimates from the previous time period `t-1`:
  * determined by the argument `previous_data`
  * objective topic location, ideological topic location are taken
  * use objective topic locations as a fixed parameter in `non_negative_factorization` from `sklearn.decomposition`
to find reasonable initial values for document locations.

Congressional speeches dataset is quite large to be run on personal computer.
Computational cluster with GPU has been used to perform the estimation.
For completion, in directory [code/create_slurms](code/create_slurms) 
we provide a code to make `.slurm` files for submitting the individual jobs.
In these scripts we carefully specify that the first session has to be initialized 
with NMF/PF and the other sessions with the results from the previous session.
A file for submitting all jobs at once uses `--dependency=singleton` to compute only one job at a time 
so that results from the previous session ready for initialization.

The output `param` directory will contain the following files:
* `document_loc.npy` - location parameter for document intensities (theta),
* `document_scale.npy` - scale parameter for document intensities (theta),
* `objective_topic_loc.npy` - location parameter for objective topics (beta),
* `objective_topic_scale.npy` - scale parameter for objective topics (beta),
* `ideological_topic_loc` - location parameter for ideological topics (eta),
* `ideological_topic_scale.npy` - scale parameter for ideological topics (eta),
* `ideal_point_loc.npy` - location parameter for ideal points,
* `ideal_point_scale.npy` - scale parameter for ideal points,
* `document_mean.npy` - log(E theta) = loc + 0.5 * var,
* `neutral_topic_mean.npy` - log(E beta) = loc + 0.5 * var,
* `negative_topic_mean.npy` - log(E beta * exp(-eta)) = (beta_loc - eta_loc) + 0.5 * (beta_var + eta_var),
* `positive_topic_mean.npy` - log(E beta * exp(+eta)) = (beta_loc + eta_loc) + 0.5 * (beta_var + eta_var),

### Post-processing the results

The directory [code/analysis](code/analysis) contains numbered 
python scripts that process the outputs from all sessions.

First file, [01_analysis](code/analysis/01_analysis.py) resaves the outputs into csv: `thetas` (log scale), `betas` (exp scale), `etas` (log scale), 
but sorted according to the importance for each topic. 
Moreover, it gathers some data from all sessions into one dataset:
* `ideal_point_speakers.csv` - ideal points for all speakers separately for each session,
* `ideal_points_all_sessions.csv` - ideal points for all speakers and all sessions combined (empty if speaker not present),
* `speeches_by_speaker.csv` - data on speeches included within the analysis including information about the speakers,
* `speeches_by_preprocessed_speakers` - number of speeches given by a speaker in each session,
* `posneg_cs.csv` - cosine similarity between positive and negative topics for each topic and session.

Python script [02_plots.py](code/analysis/02_plots.py) creates descriptive plots such as:
* boxplots of democratic and republican senator ideological positions in time,
* the evolution of the average partisanship (difference between means of democratic and republican positions) in time,
* averaged cosine similarities of positive, neutral and negative topics in time,
* wordclouds containing top 20 terms used by Republican and Democrat for each topic,
* wordclouds of top 20 neutral terms. 

R script [02_plots.R](code/analysis/02_plots.R) creates nice plots with `ggplot2`:
* boxplots of ideological positions for each session distinguished by political party,
* average partisanship in time (difference between positions of Democrats and Republicans).

File [03_influential_speeches](code/analysis/03_influential_speeches.py) finds the most influential speeches for selected senators. 
The influence is measured in terms of log-likelihood ratio test statistic for testing ideal point to be zero.

File [04_list_top_bigrams](code/analysis/04_list_top_bigrams.py) creates csv files containing top 10 terms for each topic. 
It is based on objective topics + (-1,0,1) ideological topics:
* `negative_10_bigrams.csv` - terms used by speaker of ideological position -1,
* `neutral_10_bigrams.csv` - terms used by a neutral speaker (zero ideological position),
* `positive_10_bigrams.csv` - terms used by speaker of ideological position 1.

### Simulation study

Our simulation study uses the estimated parameters from the analysis of Congressional speeches
to generate dataset of the same magnitude under different ideological positions of the speakers.

#### Generating counts + exploration

Directory [code/simulation](code/simulation) contains script
[simulate_counts](code/simulation/simulate_counts.py) 
for generating the counts and 
[explore_sampled_counts](code/simulation/explore_sampled_counts.py)
to judge the reasonability of the sampled counts. 
After many trials and errors we settled down with the following approach.

For each time period, we need the corresponding 
document intensities (theta) from file `thetas.csv`,
objective topics (beta) from file `neutral_topic_mean.npy`, (not `betas.csv` due to shuffling),
ideological topics (eta) either from `ideological_topic_loc.npy` or restored from `negative_topic_mean.npy` and `positive_topic_mean.npy`, (not `etas.csv` due to shuffling).
Since some values of eta were too extreme we decided to winsorize them into interval [-1, 1], i.e.,
values lower than -1 became -1 and
values higher than 1 became 1.
All of these are on log scale so the Poisson rates are reconstructed as
`exp(theta + beta + eta * ideal)`, where `ideal` is constructed in 4 different scenarios:
* `zero` - all ideal points are zero,
* `party` - ideal point is -0.5 for Republicans, 0.5 for Democrats and 0 for Independent senators,
* `diverge` - zero ideal points until session 100, then increase or decrease by 0.05 for each additional session towards the political party,
* `estimate` - use the estimated ideological positions for corresponding session.

Note that for the generating we already used a newer version of Tensorflow,
see [requirements_tf_TBIP](requirements_tf_TBIP.txt).

#### Estimating TVTBIP on simulated data

All computations were also performed on a computational cluster.
Hence, we provide file [simulation](code/create_slurms/simulation.py)
to create all `slurm` files needed to submit a job.
These will be saved in `slurm/simulation-scenario/` directory.
To run all of them submit `run_all_97_114.slurm`, which estimates
TBIP for a given session only after TBIP for the previous session has been estimated.

#### Post-processing results on simulated data

First, we needed to get all the estimated ideological positions into one file,
see [01_ideal_to_csv](code/simulation/01_ideal_to_csv.py).
But we noticed that the ranges of ideological positions were 
different from the ranges of ideal points on the original data.
Hence, we took the estimates of ideological topics (eta) from 
the original dataset and for the simulated one, 
computed the interquantile range (difference between Q3 and Q1) of etas
and multiplied the estimated ideological positions for simulated dataset with
`scale_coefficient = sim_eta_scl / orig_eta_scl`.
This way the summaries and plots reached comparable scales.
Rescaled ideal points were saved for each session separately into
`ideal_point_speakers_rescaledIQR.csv` and then combined into
one csv file for all sessions
`scenario_ideal_point_all_sessions_rescaledIQR.csv`
saved in common folder to all scenarios
[data/simulation](data/simulation).

Having all the estimated ideological position at one place,
we could continue with plotting the difference between
Democrat and Republican senators, see
[02_plots](code/simulation/02_plots.py). 
We have also used R and ggplot2 to produce nicer plots with
[03_plots](code/simulation/03_plots.R).





