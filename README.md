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
* unzip them into the directory [data/hein-daily/orig](data/hein-daily/orig),
* add your own file with [stopwords](data/hein-daily/stopwords.txt) (ours copied from [Keyon Vafa](https://github.com/keyonvafa/tbip)),
* run the two scripts in `code/preprocessing`: 
  * [01_supervocab.py](code/preprocessing/01_supervocab.py) to create vocabulary that spans all sessions,
  * [02_inputmatrices.py](code/preprocessing/02_inputmatrices.py) to create the count matrices and other required files.

If you wish to use TVTBIP for your own dataset, you need to keep the following structure:
* `data/your-data-name` should be the main directory to store both data and estimates from TVTBIP
* each time period should have its own subdirectory to store the data, e.g.,
  * for `hein-daily` we have [97](data/hein-daily/97), ..., [114](data/hein-daily/114)
* within the subdirectory for a time period should be two subdirectories:
  * [input](data/hein-daily/97/input) to store the inputs for the TVTBIP, which should contain:
    * `counts.npz` - the document-term matrix in sparse format,
    *  possibly `counts`+scenario+`.npz` for other data versions (simulation scenarios in our case),
    * `author_indices.npy` - vector of author indices declaring who is the author of the corresponding document,
    * `author_maptxt.` - name and surname of the speaker followed by party label in parentheses (one per line),
    * `vocabulary.txt` - list of words (one per line) that should be common to all sessions for TVTBIP,
  * [output](data/hein-daily/97/output) to store the outputs of the TVTBIP, which should be empty.
    * Could be named differently for different runs, but you need to adjust the `output` variable in code.

### Estimating TVTBIP

The estimation is performed by [estimate_session.py](code/analysis/estimate_session.py), 
which runs only one single session (time period).
The key mechanism is the initialization process.

Depending on the flag `initialization` you have three options:
* `random` - initialize model parameters completely at random, which we do not recommend,
* `NMF` - initialize the location parameters for documents and objective topics by performing Non-negative Matrix Factorisation (NMF) first, 
we use `NMF` from `sklearn.decomposition`:
  * used for the first time period (session),
* `previous` - for time period `t` initialize model parameters by estimates from the previous time period `t-1`:
  * take estimates from `output` directory of time period `t-1` 
    * objective topic location, ideological topic location
  * use objective topic locations as a fixed parameter in `non_negative_factorization` from `sklearn.decomposition`
to find reasonable initial values for document locations.

To estimate TBIP for a session we use the original implementation by [Keyon Vafa](https://github.com/keyonvafa/tbip) 
with couple of small changes. 
The TBIP is defined in [tbip.py](code/analysis/tbip.py) within the same directory. 
Both TBIP and our time-varying version are implemented in Tensorflow 1.15, 
which substantially differs from the newer versions. 
The list of the versions of used libraries is below:
 
#todo

Congressional speeches dataset is quite large to be run on personal computer.
Computational cluster with GPU has been used to perform the estimation.
For completion, in directory [code/create_slurms](code/create_slurms) 
we provide a code to make `.slurm` files for submitting the individual jobs.
In these scripts we carefully specify that the first session has to be initialized 
with NMF and the other sessions with the results from the previous session.
A file for submitting all jobs at once uses `--dependency=singleton` to compute only one job at a time 
so that results from the previous session ready for initialization.

!!! Log-Normal normal variational family used for beta and theta! Not Gamma! 

The output directory will contain the following files:
* `ideal_step_by_step.csv` - estimated ideal points for each iterative step to see the progress in estimation,
* `document_topic_mean.npy` - log(E theta) = loc + 0.5 * var,
* `neutral_topic_mean.npy` - log(E beta) = loc + 0.5 * var,
* `negative_topic_mean.npy` - log(E beta * exp(-eta)) = (beta_loc - eta_loc) + 0.5 * (beta_var + eta_var),
* `positive_topic_mean.npy` - log(E beta * exp(+eta)) = (beta_loc + eta_loc) + 0.5 * (beta_var + eta_var),
* `ideal_point_mean.npy` - location parameter for ideal points,
* `objective_topic_loc.npy` - location parameter for objective topics (beta),
* `objective_topic_scale.npy` - scale parameter for objective topics (beta),
* `ideological_topic_loc` - location parameter for ideological topics (eta),
* `ideological_topic_scale.npy` - scale parameter for ideological topics (eta),
* `loss_values.csv` - ELBO values for each iteration.


