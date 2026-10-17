# Data

Source: UCI Machine Learning Repository, Heart Disease (id 45),
https://archive.ics.uci.edu/dataset/45/heart+disease

Download and clean (writes `processed/heart_clean.csv`):

```bash
python -m heart.data                                        # Cleveland (303 rows)
python -m heart.data --sites cleveland hungarian switzerland va
```

Columns: 13 clinical features (see `src/heart/config.py`), `site`, and the binary `target` (1 = disease, from original `num > 0`). Missing values remain empty and are imputed inside the model pipeline.
