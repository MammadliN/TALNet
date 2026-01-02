# cmu-thesis

This repository contains the code for three experiments in my PhD thesis, [Polyphonic Sound Event Detection with Weak Labeling](http://www.cs.cmu.edu/~yunwang/papers/cmu-thesis.pdf), and now also a Passive Acoustic Monitoring (PAM) pipeline built on TALNet for AnuraSet and FNJV:

* Sound event detection with **presence/absence labeling** on the **[DCASE 2017 challenge](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-large-scale-sound-event-detection)** (Chapter 3.2)
* Sound event detection with **presence/absence labeling** on **[Google Audio Set](https://research.google.com/audioset/)** (Chapter 3.3)
* Sound event detection with **sequential labeling** on a subset of **[Google Audio Set](https://research.google.com/audioset/)** (Chapter 4)
* **Passive Acoustic Monitoring (PAM) WSSED** on **AnuraSet** with final testing on **FNJV**, using stratified recording-level splits and OTHERS binning for out-of-scope species (see below).

## Prerequisites

Hardware:
* A GPU
* Large storage (1 TB recommended)

Software:
* Python 2.7
* PyTorch (I used version 0.4.0a0+d3b6c5e)
* numpy, scipy, [joblib](https://pypi.org/project/joblib/)

## Quick Start

```python
# Clone the repository
git clone https://github.com/MaigoAkisame/cmu-thesis.git

# Download the data: may take up to 1 day!
cd cmu-thesis/data
./download.sh

# Train a model for the DCASE experiment using default settings
cd ../code/dcase
python train.py            # Needs to run on a GPU

# Evaluate the model at Checkpoint 25
python eval.py --ckpt=25   # Needs to run on a GPU for the first time

# Download and evaluate the TALNet model for the Audio Set experiment
cd ../audioset
./eval-TALNet.sh           # Needs to run on a GPU for the first time
```

## Organization of the Repository

### code

The `code` directory contains three sub-directories for the original experiments (`dcase`, `audioset`, and `sequential`) plus a `pam` directory for the AnuraSet/FNJV WSSED pipeline. In each experiment directory:

* `Net.py` defines the network architecture (you don't need to execute this script directly);
* `train.py` trains the network;
* `eval.py` evaluates the network's performance.

The `train.py` and `eval.py` script can take many command line arguments, which specify the architecture of the network and the hyperparameters used during training. If you encounter "out of memory" errors, a good idea is to reduce the batch size.

Some scripts that may be of special interest:

* `code/*/util_in.py`: Implements data balancing so that each minibatch contains roughly equal numbers of recordings of each event type;
* `code/sequential/ctc.py`: My implementation of connectionist temporal classification (CTC);
* `code/sequential/ctl.py`: My implementation of connectionist temporal localization (CTL).
* `code/pam/run_wssed.py`: Prepares stratified AnuraSet splits, generates bag/instance manifests for WSSED, and filters FNJV for final testing.

### PAM WSSED for AnuraSet and FNJV

The PAM workflow replaces the original AudioSet data readers with AnuraSet/FNJV-aware loaders. Key choices:

* Target species codes: `DENMIN`, `LEPLAT`, `PHYCUV`, `SPHSUR`, `SCIPER`, `BOABIS`, `BOAFAB`, `LEPPOD`, `PHYALB`; all other species map to `OTHERS`.
* AnuraSet labels are canonicalized by stripping suffixes (e.g., `BOABIS_L` → `BOABIS`) before the target/OTHERS mapping.
* Recording-level stratified splits (train/val/test) are derived from AnuraSet strong labels so that segments from the same recording never leak across splits.
* FNJV is only used for final testing; rows with `Code == IGNORE` or codes outside the target list are excluded.
* Segment generation supports instances of 1/3/5/10/15/30/60 seconds and bags of 60/120/300/600 seconds; every experiment iterates the full grid in increasing bag length.

TALNet is used for WSSED on the PAM manifests with a 10-class head (nine target species plus `OTHERS`). The `code/pam/training.py` CLI wraps the original TALNet architecture and trains on the generated bag manifests:

```bash
cd code/pam
# Example for instance=1s, bag=60s, reading audio from /path/to/wavs
python training.py \
  --manifest-root ../../workspace/pam \
  --config instance1_bag60 \
  --audio-root /path/to/wavs \
  --epochs 10 --batch-size 16
```

This will save TALNet weights (`pam_talnet.pt`) and evaluation metrics (`pam_talnet_metrics.json`) inside the configuration directory after optionally validating on AnuraSet and testing on both AnuraSet and FNJV splits when available.

To prepare manifests for all experiments:

```bash
cd code/pam
python run_wssed.py \
  --anuraset-root /path/to/AnuraSet/csvs \
  --fnjv-metadata /path/to/FNJV/458/metadata_filtered_filled.csv \
  --output-dir ../../workspace/pam \
  --val-size 0.1 --test-size 0.2
```

The script writes normalized annotations, per-recording label counts, per-split segment CSVs, and a manifest describing the chosen instance/bag configuration. Set `--val-size 0` to skip validation splits when performing training without early stopping.

### data

The script `data/download.sh` will download and extract the following three archives in the `data` directory:

* [dcase.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/dcase.tgz) (4.9 GB)
* [audioset.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/audioset.tgz) (341 GB)
* [sequential.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/sequential.tgz) (63 GB)

These archives contain Matlab data files (with the `.mat` extension) that store the filterbank features and ground truth labels. They can be loaded with the `scipy.io.loadmat` function in Python. Each Matlab file contains three matrices:

* `feat`: Filterbank features, a float32 array of shape (n, 400, 64) (n recordings, 400 frames, 64 frequency bins);
* `labels`:
  * Presence/absence labeling, a boolean array of shape (n, m) (n recordings, m event types), or
  * Strong labelng, a boolean array of shape (n, 100, m) (n recordings, 100 frames, m event types);
* `hashes`: A character array of size (n, 11), containing the YouTube hash IDs of the recordings.

Training recordings are organized by class (so data balancing can be done easily), and each Matlab file contains up to 101 recordings. Validation and test/evaluation recordings are stored in Matlab files that contain up to 500 recordings each.

Because the data is so huge, I do not provide the code for downloading the raw data, extracting features, and organizing the features and labels into Matlab data files. The whole process took me more than a month and endless babysitting!

### workspace

The training logs, trained models, predictions on the test/evaluation recordings, and evaluation results will be generated in this directory. The sub-directory names will reflect the network architecture and hyperparameters for training.

The script `code/audioset/eval-TALNet.py` will download the TALNet model and store it at `workspace/audioset/TALNet/model/TALNet.pt`. At the time of my graduation (October 2018), this is the best model that can both classify and localize sound events on Google Audio Set.

## Citing

If you use this code in your research, please cite my PhD thesis:

* Yun Wang, "Polyphonic sound event detection with weak labeling", PhD thesis, Carnegie Mellon University, Oct. 2018.

and/or the following publications:

* Yun Wang, Juncheng Li and Florian Metze, "A comparison of five multiple instance learning pooling functions for sound event detection with weak labeling," arXiv e-prints, Oct. 2018. [Online]. Available: <http://arxiv.org/abs/1810.09050>.
* Yun Wang and Florian Metze, "Connectionist temporal localization for sound event detection with sequential labeling," arXiv e-prints, Oct. 2018. [Online]. Available: <http://arxiv.org/abs/1810.09052>.
