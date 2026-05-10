# Carbon-Aware HPC Scheduling

This project explores a simple carbon-aware scheduling approach for HPC jobs.

The work is split into two steps:

1. train a model to estimate the energy consumption of PM100 jobs;
2. use the predicted energy in a scheduler that tries to run jobs during lower-carbon-intensity time windows.

The scheduler is compared against:

- `fcfs`: first-come first-served;
- `random`: random feasible delay;
- `carbon_aware`: chooses the feasible start time with the lowest average carbon intensity within a 24-hour delay window.

## Project Structure

```text
project_rebuild/
  config/                  hardware assumptions
  data/                    dataset instructions
  notebooks/               result exploration notebooks
  scripts/                 runnable pipeline scripts
  src/hpc_carbon_scheduler core Python modules
  results/                 generated CSV files and plots
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

Run the energy prediction step:

```bash
python scripts/01_run_carbon_prediction.py --max-rows 60000
```

Then run the scheduling simulation:

```bash
python scripts/02_run_scheduler.py --num-weeks 0 --max-jobs 1200 --split all
```

`--num-weeks 0` means that all available weeks in the prediction file are used.

## Results

Energy prediction with `--max-rows 60000`:

| Metric | Value |
| --- | ---: |
| Valid jobs | 58,846 |
| Train rows | 44,134 |
| Test rows | 14,712 |
| Energy MAE | 6.76 kWh |
| Energy R2 | 0.49 |
| Emissions MAE | 2.46 kg CO2 |

Scheduling simulation over 10 real weeks:

| Policy | Jobs | Emissions | Reduction vs FCFS | Avg. wait |
| --- | ---: | ---: | ---: | ---: |
| FCFS | 12,000 | 27,365 kg CO2 | 0.00% | 0.00 h |
| Random delay | 12,000 | 27,079 kg CO2 | 1.04% | 11.74 h |
| Carbon-aware | 12,000 | 25,965 kg CO2 | 5.11% | 12.40 h |

Carbon-aware savings compared with FCFS:

| Statistic | Value |
| --- | ---: |
| Total CO2 saved | 1,399.21 kg CO2 |
| Average CO2 saved per week | 139.92 kg CO2 |
| Minimum weekly CO2 saved | 44.61 kg CO2 |
| Maximum weekly CO2 saved | 431.22 kg CO2 |
| Total weighted reduction | 5.11% |
| Average weekly reduction | 7.46% |
| Minimum weekly reduction | 3.07% |
| Maximum weekly reduction | 14.41% |

The weighted reduction is computed on total emissions across all simulated jobs. The weekly average is the simple average of the 10 weekly percentage reductions, so it gives more visibility to week-by-week variability.

## Notes

- Carbon intensity is synthetic, not real grid data.
- `num_nodes_req` is treated as requested processors, following the project assumptions.
- The simulated machine capacity is 15,680 processors.
- Memory is documented but not enforced as a hard scheduling constraint.

## Suggested Repository Name

```text
carbon-aware-hpc-scheduling
```
