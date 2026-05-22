# Carbon-Aware HPC Scheduling

This project studies whether HPC jobs can be scheduled in a more carbon-aware way without changing the workload itself. The idea is to first estimate how much energy each job is likely to consume, then use that estimate inside a scheduler that can delay jobs toward time windows with lower carbon intensity.

The project is based on the PM100 HPC trace. Since the trace does not provide a real electricity-grid carbon signal, the carbon intensity used here is synthetic and time-dependent. The simulated profile follows a plausible daily and seasonal pattern inspired by the area of Bologna, but it is not measured grid data from the Municipality of Bologna. This makes the project a controlled simulation rather than a production-ready scheduler.

Research question:

> Can predicted job energy consumption be used to reduce simulated HPC carbon emissions by delaying jobs toward lower-carbon periods?

The workflow has two phases:

1. build an energy-prediction dataset from the HPC trace;
2. compare multiple regression models on a chronological validation split;
3. select the best model and generate energy predictions for all valid jobs;
4. simulate different scheduling policies over multiple real weeks of the trace;
5. measure emissions, waiting time, and week-by-week variability.

The scheduling phase compares:

- `fcfs`: first-come first-served;
- `random`: random feasible delay;
- `carbon_aware`: chooses the feasible start time with the lowest estimated carbon cost within a 24-hour delay window.

## Project Structure

```text
project_rebuild/
  config/                  hardware assumptions
  data/                    dataset instructions
  notebooks/               model and scheduler result notebooks
  scripts/                 runnable pipeline scripts
  src/hpc_carbon_scheduler core Python modules
  results/                 generated CSV files and plots for both scheduler runs
```

Main scripts:

```text
scripts/01_run_carbon_prediction.py
scripts/02_run_scheduler.py
```

## Dataset

The original PM100 parquet dataset is not included because it is too large for a normal GitHub repository.

To reproduce the full pipeline, place it here:

```text
data/job_table.parquet
```

Alternatively, pass a custom path:

```bash
python scripts/01_run_carbon_prediction.py --input path/to/job_table.parquet
```

The repository already includes generated results and plots, so it can be inspected without the raw dataset.

## Setup

From inside `project_rebuild`:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Reproduce the Results

Run the energy prediction/model-selection step:

```bash
python scripts/01_run_carbon_prediction.py --max-rows 60000
```

Then run the controlled scheduling simulation with at most 1,200 jobs per week:

```bash
python scripts/02_run_scheduler.py --num-weeks 0 --max-jobs 1200 --split all --output-dir results/scheduler_capped_1200
```

Run the full weekly workload simulation:

```bash
python scripts/02_run_scheduler.py --num-weeks 0 --max-jobs 0 --split all --output-dir results/scheduler_full_workload
```

`--num-weeks 0` means that all available weeks in the prediction file are used. `--max-jobs 0` disables the weekly cap and schedules every available job in each selected week.

The scheduler notebooks are separated by experiment:

- `notebooks/02_scheduler_capped_1200.ipynb`;
- `notebooks/03_scheduler_full_workload.ipynb`.

## Results

Energy prediction with `--max-rows 60000`.

The first phase compares four models on the validation split and selects the one with the highest validation energy R2:

| Model | Validation MAE | Validation R2 |
| --- | ---: | ---: |
| HistGradientBoosting | 6.41 kWh | 0.56 |
| Random Forest | 6.65 kWh | 0.52 |
| Extra Trees | 7.18 kWh | 0.37 |
| Ridge Regression | 9.28 kWh | -0.10 |

Selected model: `hist_gradient_boosting`.

| Metric | Value |
| --- | ---: |
| Valid jobs | 58,846 |
| Train rows | 35,307 |
| Validation rows | 11,769 |
| Test rows | 11,770 |
| Test energy MAE | 5.81 kWh |
| Test energy R2 | 0.34 |
| Test emissions MAE | 2.13 kg CO2 |

Full workload scheduling simulation over 10 real weeks:

| Policy | Jobs | Emissions | Reduction vs FCFS | Avg. wait |
| --- | ---: | ---: | ---: | ---: |
| FCFS | 58,846 | 115,771 kg CO2 | 0.00% | 0.00 h |
| Random delay | 58,846 | 115,625 kg CO2 | 0.13% | 11.74 h |
| Carbon-aware | 58,846 | 109,469 kg CO2 | 5.44% | 11.55 h |

Full workload carbon-aware savings compared with FCFS:

| Statistic | Value |
| --- | ---: |
| Total CO2 saved | 6,302.60 kg CO2 |
| Average CO2 saved per week | 630.26 kg CO2 |
| Minimum weekly CO2 saved | 153.19 kg CO2 |
| Maximum weekly CO2 saved | 1,132.84 kg CO2 |
| Total weighted reduction | 5.44% |
| Average weekly reduction | 6.56% |
| Minimum weekly reduction | 4.16% |
| Maximum weekly reduction | 11.26% |

Controlled capped simulation over the same 10 weeks, using at most 1,200 jobs per week:

| Policy | Jobs | Emissions | Reduction vs FCFS | Avg. wait |
| --- | ---: | ---: | ---: | ---: |
| FCFS | 12,000 | 27,365 kg CO2 | 0.00% | 0.00 h |
| Random delay | 12,000 | 27,079 kg CO2 | 1.04% | 11.74 h |
| Carbon-aware | 12,000 | 25,966 kg CO2 | 5.11% | 12.36 h |

Capped-run carbon-aware savings compared with FCFS:

| Statistic | Value |
| --- | ---: |
| Total CO2 saved | 1,399.15 kg CO2 |
| Average CO2 saved per week | 139.92 kg CO2 |
| Minimum weekly CO2 saved | 44.61 kg CO2 |
| Maximum weekly CO2 saved | 431.22 kg CO2 |
| Total weighted reduction | 5.11% |
| Average weekly reduction | 7.46% |
| Minimum weekly reduction | 3.07% |
| Maximum weekly reduction | 14.41% |

The full workload run is the direct weekly workload simulation. The capped run is useful as a controlled comparison with the same maximum weekly sample size. In both tables, the weighted reduction is computed on total emissions across all simulated jobs, while the weekly average is the simple average of the 10 weekly percentage reductions.

## Limitations and Assumptions

- Carbon intensity is synthetic. It follows a plausible Bologna-area pattern, but it is not measured grid data.
- The scheduler uses `pred_energy_kwh` generated by the selected model in phase 1.
- The published scheduler runs use `--split all` to study all available weekly windows; use `--split test` for a stricter simulation limited to prediction-test jobs.
- `num_nodes_req` is treated as requested processors, following the project assumptions.
- The simulated machine capacity is 15,680 processors.
- Memory is included in the energy-prediction features but is not enforced as a hard scheduling constraint.
