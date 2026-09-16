# LaLiga Match Forecasting

Reproducible data preparation and baseline forecasting for LaLiga football
matches. The project treats the team as a changing time-series state rather
than as a permanent club statistic.

## Current status

Version **0.1.0** is the results-and-basic-statistics foundation. It downloads
historical LaLiga SP1 CSV files from Football-Data.co.uk, creates chronological
pre-match features, and evaluates a small multinomial baseline. It does not yet
claim to be player-aware: lineups, minutes, transfers, managers, injuries, and
player form are planned enrichment layers.

The generated data is deliberately not committed to Git. The source files are
downloaded reproducibly by the build module and the Hugging Face publication
cell.

## Repository layout

```text
src/laliga_forecasting/build_dataset.py   # download, normalize, feature build, Hub upload
src/laliga_forecasting/train_baseline.py  # chronological baseline evaluation
requirements-colab.txt                    # Colab runtime dependencies
```

## Run locally

Use the project Python environment:

```bash
~/.venv/bin/python -m pip install -e .
PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.build_dataset --output-dir outputs/v0.1.0
PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.train_baseline \
  --input outputs/v0.1.0/pre_match_forecasting.parquet \
  --output outputs/v0.1.0/baseline_metrics.json
```

The local build never uploads anything. The source publisher's terms and
attribution requirements apply; verify redistribution rights before treating
any generated copy as an openly licensed dataset.

## Run in Google Colab

The reproducible Colab workflow uses the notebook's `HF_TOKEN` secret. It does
not call `input()`, `getpass()`, or print the token.

```python
import os
os.chdir('/content')
!pip install -q -r https://raw.githubusercontent.com/EF-Code/laliga-match-forecasting/main/requirements-colab.txt
from pathlib import Path
import shutil

repo_dir = Path('/content/laliga-match-forecasting')
if repo_dir.exists():
    shutil.rmtree(repo_dir)
!git clone -q https://github.com/EF-Code/laliga-match-forecasting.git /content/laliga-match-forecasting
os.chdir('/content/laliga-match-forecasting')

import sys
import importlib
from google.colab import userdata
importlib.invalidate_caches()
sys.path.insert(0, '/content/laliga-match-forecasting/src')
import laliga_forecasting.build_dataset as builder

hf_token = userdata.get('HF_TOKEN')
if not hf_token:
    raise RuntimeError('Add a write-capable HF_TOKEN secret in Colab and enable notebook access')
output_dir = Path('/content/laliga-output')
matches = builder.fetch_all_seasons()
pre_match, observations, team_stats = builder.build_pre_match_dataset(matches)
paths = builder.write_outputs(output_dir, pre_match, observations, team_stats)
repo_id = builder.publish_to_hub(output_dir, pre_match, paths, token=hf_token)
del hf_token

!PYTHONPATH=src python -m laliga_forecasting.train_baseline --input /content/laliga-output/pre_match_forecasting.parquet --output /content/laliga-output/baseline_metrics.json
```

Add a Colab Secret named `HF_TOKEN` and enable notebook access before running
the publication cell. The token is read in the notebook kernel, passed in
memory to the uploader, and never prompted for, printed, or written to disk.
The code resolves the HF account from that secret and publishes to
`<account>/laliga-football-match-forecasting`.

## Data boundary

`pre_match_forecasting.parquet` contains features calculated from earlier
matches and the current match's labels. The separate observation artifacts
contain post-match statistics and must not be joined back into pre-match
features without respecting the match-time cutoff.

Future releases will add player and roster snapshots only after a consistent,
licensed source is selected. A model should be evaluated chronologically and
compared with simple dynamic baselines before any language model is introduced.

## Source

[Football-Data.co.uk Spain data](https://www.football-data.co.uk/spainm.php)
