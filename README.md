# Carbon-Aware HPC Scheduling with Energy Prediction

This project simulates a carbon-aware scheduler for HPC jobs. The work is split into two phases:

1. predict each job's energy consumption from the information available in the PM100 trace;
2. use that prediction to shift jobs toward time windows with lower carbon intensity, then compare the carbon-aware scheduler against two baselines.

The project uses the PM100 HPC job trace and a synthetic carbon-intensity curve. The original dataset is not included in this repository because it is large, but the code, notebooks, generated results, and documentation are ready to inspect and reproduce.

## Goal

Supercomputers consume a large amount of energy. The environmental impact of a job depends not only on how much energy it consumes, but also on when it runs. For the same amount of energy, a job executed during a lower-carbon-intensity period emits less CO2.

This project experimentally evaluates whether a scheduling policy that uses predicted job energy can reduce emissions compared with a standard FCFS baseline.

## Repository Structure

```text
project_rebuild/
  config/
    hardware_assumptions.json
  data/
    README.md
  notebooks/
    01_carbon_prediction.ipynb
    02_scheduler_simulation.ipynb
  scripts/
    01_run_carbon_prediction.py
    02_run_scheduler.py
  src/hpc_carbon_scheduler/
    carbon_model.py
    scheduler.py
    plotting.py
  results/
    carbon_prediction/
    scheduler/
  requirements.txt
```

### Main Files

| File | Purpose |
| --- | --- |
| `scripts/01_run_carbon_prediction.py` | Runs the prediction phase: data loading, feature engineering, model training, metrics, and plots. |
| `scripts/02_run_scheduler.py` | Runs the multi-week scheduling simulation. |
| `src/hpc_carbon_scheduler/carbon_model.py` | Contains data preparation, energy/emissions targets, chronological split, and the Random Forest model. |
| `src/hpc_carbon_scheduler/scheduler.py` | Contains the scheduler logic, weekly-window selection, and capacity checks. |
| `src/hpc_carbon_scheduler/plotting.py` | Generates the plots used by the notebooks and result folders. |
| `notebooks/01_carbon_prediction.ipynb` | Notebook for inspecting and explaining the prediction results. |
| `notebooks/02_scheduler_simulation.ipynb` | Notebook for inspecting and explaining the scheduling results. |

## Phase 1: Energy and Emissions Prediction

The first phase loads completed jobs, filters invalid records, and computes actual energy from the power samples available in the dataset:

```text
energy_kWh = avg_power_W * run_time_seconds / (3600 * 1000)
```

It then generates a synthetic carbon-intensity signal in `gCO2/kWh` and computes emissions:

```text
emissions_kg = energy_kWh * gCO2_per_kWh / 1000
```

The predictive model is a `RandomForestRegressor` trained on log-transformed energy. The split is chronological: the model is trained on the earlier part of the trace and evaluated on the later part, avoiding random mixing between past and future.

The features include:

- requested resources (`num_nodes_req`, `num_cores_req`, `num_gpus_req`, `mem_req`);
- time limit (`time_limit`);
- task count and cores per task;
- submission-time features (`hour`, `weekday`, `month`).

The actual runtime is not used as an input feature, to avoid leakage.

## Phase 2: Carbon-Aware Scheduling

The second phase reads `results/carbon_prediction/job_predictions.csv` and simulates three scheduling policies:

| Policy | Description |
| --- | --- |
| `fcfs` | First-Come First-Served: each job starts as soon as capacity is available. |
| `random` | Baseline that chooses a random feasible delay within the allowed window. |
| `carbon_aware` | Searches within a 24-hour window and chooses the feasible start time with the lowest average carbon intensity. |

The scheduler respects:

- job release time, so a job cannot start before submission;
- job duration represented on discrete time slots;
- parallel system capacity;
- maximum delay;
- identical weekly job sets for all policies.

## Hardware Assumptions

The experimental notes specify:

- 980 available nodes;
- 16 processors per node;
- 256000 memory units per node;
- total capacity: `980 * 16 = 15680` processors.

In the PM100 dataset, the column `num_nodes_req` is treated as the number of requested processors despite its misleading name. The simulation therefore checks that the sum of requested processors from concurrently running jobs stays below 15,680.

## Dataset

The original dataset is not committed to this repository. By default, the scripts expect it at:

```text
data/job_table.parquet
```

If the file is stored elsewhere, pass the path explicitly with `--input`:

```bash
python scripts/01_run_carbon_prediction.py --input path/to/job_table.parquet --max-rows 60000
```

The final run uses the first 60,000 records sorted chronologically. After filtering, 58,846 valid jobs remain across 10 real weeks.

The repository is therefore standalone in terms of code, notebooks, documentation, and generated results. The only external artifact is the raw PM100 parquet dataset, which should be copied into `data/job_table.parquet` if you want to regenerate the pipeline from scratch.

## Installation

From a terminal, enter the project directory:

```bash
cd project_rebuild
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Full Reproduction

Run the energy prediction phase first:

```bash
python scripts/01_run_carbon_prediction.py --max-rows 60000
```

This generates:

```text
results/carbon_prediction/metrics.csv
results/carbon_prediction/job_predictions.csv
results/carbon_prediction/feature_importance.csv
results/carbon_prediction/*.png
```

Then run the scheduler:

```bash
python scripts/02_run_scheduler.py --num-weeks 0 --max-jobs 1200 --split all
```

Main parameter meanings:

| Parameter | Meaning |
| --- | --- |
| `--num-weeks 0` | Use every available week in the predictions file. |
| `--max-jobs 1200` | Limit each simulated week to 1,200 jobs, making weekly comparisons easier. |
| `--split all` | Use both train and test prediction rows to cover more weeks in the scheduling simulation. |
| `--split test` | Use only test-set jobs. This is more strictly out-of-sample but covers fewer weeks. |
| `--max-delay-hours 24` | Maximum delay considered by the random and carbon-aware policies. |

The scheduler generates:

```text
results/scheduler/scheduler_metrics.csv
results/scheduler/scheduler_weekly_metrics.csv
results/scheduler/schedules.csv
results/scheduler/skipped_jobs.csv
results/scheduler/*.png
```

## Final Run Results

### Energy Prediction

With `--max-rows 60000`:

| Metric | Value |
| --- | ---: |
| Valid jobs | 58,846 |
| Training rows | 44,134 |
| Test rows | 14,712 |
| Energy MAE | 6.76 kWh |
| Energy RMSE | 35.42 kWh |
| Energy R2 | 0.49 |
| Synthetic emissions MAE | 2.46 kg CO2 |
| Synthetic emissions R2 | 0.47 |

### Multi-Week Scheduler

The final simulation uses 10 real weeks, with 1,200 jobs per week:

| Policy | Scheduled jobs | Skipped jobs | Emissions | Reduction vs FCFS | Average wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCFS | 12,000 | 0 | 27,365 kg CO2 | 0.00% | 0.00 h |
| Random delay | 12,000 | 0 | 27,079 kg CO2 | 1.04% | 11.74 h |
| Carbon-aware | 12,000 | 0 | 25,965 kg CO2 | 5.11% | 12.40 h |

The carbon-aware scheduler saves about 1,399 kg CO2 compared with FCFS. The weekly reduction is not constant: the unweighted weekly average is 7.46%, with a minimum of 3.07% and a maximum of 14.41%.

## Main Plots

The plots are already generated in `results/`.

Energy prediction:

- `results/carbon_prediction/energy_actual_vs_predicted.png`
- `results/carbon_prediction/feature_importance.png`
- `results/carbon_prediction/emissions_error_histogram.png`

Scheduler:

- `results/scheduler/scheduler_weekly_emissions.png`
- `results/scheduler/scheduler_weekly_reduction_pct.png`
- `results/scheduler/scheduler_weekly_energy.png`
- `results/scheduler/scheduler_weekly_wait.png`
- `results/scheduler/scheduler_weekly_avg_carbon.png`
- `results/scheduler/scheduler_emissions_comparison.png`
- `results/scheduler/scheduler_wait_comparison.png`

## Notebooks

For exploratory result inspection:

```text
notebooks/01_carbon_prediction.ipynb
notebooks/02_scheduler_simulation.ipynb
```

The notebooks mainly read the generated CSV files and plots, so they are intended as analysis and presentation views rather than the primary pipeline implementation.

## Limitations

- Carbon intensity is synthetic, not taken from real grid data.
- Memory is documented but not enforced as a hard scheduling constraint.
- The final scheduling simulation uses `--split all` to cover 10 weeks. Running only with `--split test` is stricter but covers fewer weeks.
- Carbon-aware scheduling reduces emissions by introducing waiting time, so the result should be interpreted as a CO2 vs quality-of-service trade-off.

## Suggested GitHub Title

**Carbon-Aware HPC Scheduling with Energy Prediction**

Suggested repository name:

```text
carbon-aware-hpc-scheduling
```
