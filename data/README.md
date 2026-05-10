# Dataset placement

The rebuild scripts expect the original PM100 job trace at:

```text
../project/job_table.parquet
```

If you want to run the project from a standalone clone, place the dataset locally and pass its path explicitly:

```bash
python scripts/01_run_carbon_prediction.py --input path/to/job_table.parquet --max-rows 60000
python scripts/02_run_scheduler.py --num-weeks 0 --max-jobs 1200 --split all
```

The dataset itself is not committed because it is large and should be distributed separately.
