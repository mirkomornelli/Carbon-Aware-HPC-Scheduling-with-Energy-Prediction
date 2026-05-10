# Dataset

The original PM100 parquet dataset is not included in this repository because it is too large for a normal GitHub upload.

To reproduce the full pipeline, place the dataset here:

```text
data/job_table.parquet
```

The prediction script also accepts a custom path:

```bash
python scripts/01_run_carbon_prediction.py --input path/to/job_table.parquet
```

