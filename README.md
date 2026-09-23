# Homotopic information gain

Code for our paper [Homotopic Information Gain for Sparse Active Target Tracking](https://ieeexplore.ieee.org/abstract/document/11427336), which introduces homotopic information gain, the information gained from measurements regarding a pedestrian's *homotopy class*. Performing information gathering/active target tracking with this measure allows for the multi-modal nature of pedestrian trajectories to be considered during planning, resulting in impressive trajectory estimation with very sparse measurement sets.

This repository is a minimal working example on the ATC shopping-mall pedestrian dataset. It includes a trained model, the train/test trajectories, and the scripts used to process the raw ATC data and train the model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11. Keep the version pins in `requirements.txt`: the code needs NumPy 1.x and matplotlib < 3.9.

## Repository layout

```
run_single_target_eval.py            Run the planning experiment on ATC test trajectories
run_orienteering_mcts_experiment.py  One experiment: predict, plan with MCTS, measure, update
evaluate_single_target_results.py    Summarise experiment results
process_atc_data.py                  Raw ATC CSV -> per-person trajectories
reclean_retrain.py                   Clean trajectories, sort by homotopy signature, split train/test
atc_model_training.py                Train the VMMP and per-signature GMMs
vmmp_gmm/                            Homotopy signatures, VMMP, trajectory GMMs (prediction model)
ipp/                                 Measurement model, information-gain rewards, MCTS planner, robot
map_files/                           ATC mall occupancy map and boundary
data/                                ATC train/test trajectories and trained model
```

The included data:

| File | Contents |
|---|---|
| `data/atc_train_traces_tro` | Training trajectories, grouped by homotopy signature |
| `data/atc_test_traces_tro` | Test trajectories |
| `data/atc_trained_model_tro` | Trained VMMP and per-signature GMMs |

The trained model is a pickle containing `vmmp_gmm` objects, so keep the module names and layout unchanged or it will not load.

## Running the example

**Run all scripts from the repository root.** 

```bash
mkdir -p results results_figs
python run_single_target_eval.py 0 kl atc results
#                                ^ start index, reward, file suffix, save folder
```

Possible reward arguments are `kl` or `ent`. To run with the homotopic information gain reward, select `kl`, referring to the KL divergence between homotopic beliefs. To run with the approximated entropy of a GMM, select `ent`.

This runs 100 test trajectories, starting at the index you give (here 0–99). Each takes on the order of a minute. For each trajectory it saves a result pickle to `results/kl_atc_<index>`, and a figure per planning step to `results_figs/`. **Create both folders before running**, or the script will fail at the first figure.


### Evaluating results

```bash
python evaluate_single_target_results.py results atc atc_test_traces_tro atc_trained_model_tro 0 100 eval_summary
#                                        ^ results folder, file suffix, test data, model, first index, end index (exclusive), output name
```

This reads the `kl_atc_*` and `ent_atc_*` results in the index range, and writes a summary for each reward to `results/eval_summary`: average displacement error, KL divergence, mutual information, number of measurements and failures.

### Retraining the model

```bash
python atc_model_training.py
```

This reads `data/atc_train_traces_tro` and overwrites `data/atc_trained_model_tro`. GMM fitting is not seeded, so a retrained model will differ slightly from the included one.

## Processing the raw ATC data

The raw data is not included. Download the ATC CSVs for 2012-10-24 and 2012-10-28 into `atc_mall/`, then run:

1. `python process_atc_data.py`: converts a raw CSV into `atc_mall/processed_data_<date>`. The input date is hardcoded, so edit it and run once per day.
2. `python reclean_retrain.py`: cleans the trajectories, sorts them by homotopy signature, and writes `data/atc_train_traces_tro` and `data/atc_test_traces_tro`.
3. `python atc_model_training.py`: trains the model, as above.
