# LaLiga Match Forecasting

Reproducible data preparation and baseline forecasting for LaLiga football
matches. The project treats the team as a changing time-series state rather
than as a permanent club statistic.

## Current status

Version **0.2.0** keeps the results-and-basic-statistics foundation and adds a
separate StatsBomb Open Data enrichment bundle. It downloads the available male
La Liga matches from the 2014/15 through 2020/21 entries in the StatsBomb
competition catalog, then normalizes match metadata, lineups, event records,
players, and manager-team assignments. The coverage artifact records the
actual match count for each season; these are selected open-data records, not a
claim that every season is complete.

The StatsBomb artifacts are observed-match data and are not yet used as
pre-match forecasting features. Transfers, injuries, and player-form features
remain future enrichment layers.

The generated data is deliberately not committed to Git. The source files are
downloaded reproducibly by the build module and the Hugging Face publication
cell.

## Repository layout

```text
src/laliga_forecasting/build_dataset.py   # download, normalize, feature build, Hub upload
src/laliga_forecasting/statsbomb_open_data.py # StatsBomb La Liga JSON ingestion and normalization
src/laliga_forecasting/train_baseline.py  # chronological baseline evaluation
requirements-colab.txt                    # Colab runtime dependencies
```

## Run locally

Use the project Python environment:

```bash
~/.venv/bin/python -m pip install -e .
PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.build_dataset --output-dir outputs/v0.2.0 --with-statsbomb
PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.train_baseline \
  --input outputs/v0.2.0/pre_match_forecasting.parquet \
  --output outputs/v0.2.0/baseline_metrics.json
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
statsbomb_bundle = builder.build_statsbomb_bundle(output_dir / 'statsbomb-cache')
paths = builder.write_outputs(output_dir, pre_match, observations, team_stats, statsbomb_bundle)
repo_id = builder.publish_to_hub(output_dir, pre_match, paths, token=hf_token)
del hf_token

!PYTHONPATH=src python -m laliga_forecasting.train_baseline --input /content/laliga-output/pre_match_forecasting.parquet --output /content/laliga-output/baseline_metrics.json
```

Add a Colab Secret named `HF_TOKEN` and enable notebook access before running
the publication cell. The token is read in the notebook kernel, passed in
memory to the uploader, and never prompted for, printed, or written to disk.
The code resolves the HF account from that secret and publishes to
`<account>/laliga-football-match-forecasting`.

## Published artifacts

The Hub dataset contains the existing chronological `pre_match_forecasting`
splits and basic-statistics artifacts, plus these StatsBomb artifacts:

- `statsbomb_matches.parquet` — match metadata, score fields, and manager JSON references.
- `statsbomb_lineups.parquet` — one row per listed player/team/match, with starter and appearance flags, position intervals, cards, and an estimated played-minute field.
- `statsbomb_events.parquet` — one row per event with common time/team/player fields and the complete nested event in `event_json`.
- `statsbomb_players.parquet` — deduplicated player dimension keyed by `player_id`.
- `statsbomb_managers.parquet` — manager-team-match assignments.
- `statsbomb_coverage.parquet` — actual season coverage, source URLs, timestamps, and load status.

The StatsBomb Open Data [README](https://github.com/hudl/open-data/blob/master/README.md)
describes selected data as freely available for public research and football
analytics and asks published research, analysis, or insights to credit
StatsBomb and use its logo. The dataset card retains that attribution note;
review the source terms before redistributing the raw source files.

## Data boundary

`pre_match_forecasting.parquet` contains features calculated from earlier
matches and the current match's labels. The separate observation artifacts
contain post-match statistics and must not be joined back into pre-match
features without respecting the match-time cutoff.

The StatsBomb lineups and events are post-match observations. A forecasting
feature may use them only after applying a match-time cutoff; do not join the
final lineup or event artifact directly into a pre-match row. A model should be
evaluated chronologically and compared with simple dynamic baselines before
any language model is introduced.

## Source

[Football-Data.co.uk Spain data](https://www.football-data.co.uk/spainm.php)
