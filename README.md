# LaLiga Match Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/NewSonnet/laliga-football-match-forecasting)

Reproducible match data and leakage-safe baselines for time-aware LaLiga
forecasting.

This project models a club as a changing time-series state. It keeps results,
observed match data, and optional player-level enrichment separate so that
rosters, managers, and historical performance can change from season to
season without being treated as permanent club attributes.

The forecasting path is tabular and time-series based. The repository does not
train a language model; the current baseline is a chronological multiclass
classifier, with room for stronger statistical and machine-learning models.

## Current release: v0.2.0

The published [Hugging Face dataset](https://huggingface.co/datasets/NewSonnet/laliga-football-match-forecasting)
contains a base results-and-statistics layer and a separate StatsBomb Open Data
enrichment layer.

### Published snapshot

These counts were read back from the public v0.2.0 Hub release. Upstream files
can change, so reruns should be identified by their source URLs, timestamps,
and coverage artifact.

| Artifact | Purpose | Rows |
| --- | --- | ---: |
| `pre_match_forecasting` | Pre-kickoff features and full-time labels | 4,560 |
| `match_observations` | Observed results and post-match basic statistics | 4,560 |
| `team_match_stats` | Long-form observed team-match statistics | 9,120 |
| `statsbomb_matches` | StatsBomb match metadata and score fields | 590 |
| `statsbomb_lineups` | Player-team-match lineup observations | 21,512 |
| `statsbomb_events` | Normalized StatsBomb event records | 2,099,452 |
| `statsbomb_players` | Deduplicated player dimension | 1,428 |
| `statsbomb_managers` | Manager-team-match assignments | 1,186 |
| `statsbomb_coverage` | Season-level coverage and load status | 7 |

The main forecasting table has season-based splits:

| Split | Seasons | Rows |
| --- | --- | ---: |
| `train` | 2014/15–2023/24 | 3,800 |
| `validation` | 2024/25 | 380 |
| `test` | 2025/26 | 380 |

## What is included

### Main forecasting table

`pre_match_forecasting` contains one row per match. Its target is the full-time
result: `H` (home win), `D` (draw), or `A` (away win). It also contains the
home and away goal labels for downstream regression or probabilistic modeling.

The current feature set is calculated sequentially and includes:

- Elo ratings immediately before kickoff;
- matches seen by each team;
- last-five-match averages for points, goals, shots, shots on target, and
  corners;
- last-five win rates; and
- rest days since each team's previous match.

The exact field definitions are in
[`artifacts/data_dictionary.json`](https://huggingface.co/datasets/NewSonnet/laliga-football-match-forecasting/blob/main/artifacts/data_dictionary.json).

### Historical enrichment tables

The StatsBomb bundle is published under `artifacts/` on the Hub:

| Path | Contents |
| --- | --- |
| `artifacts/statsbomb_matches.parquet` | Available male LaLiga match metadata, scores, teams, and manager JSON references. |
| `artifacts/statsbomb_lineups.parquet` | One row per listed player, team, and match, including starter/appearance flags, position intervals, cards, and an estimated played-minute field. |
| `artifacts/statsbomb_events.parquet` | One row per event with common time, team, player, and position fields plus the complete nested event in `event_json`. |
| `artifacts/statsbomb_players.parquet` | Deduplicated player dimension keyed by StatsBomb `player_id`. |
| `artifacts/statsbomb_managers.parquet` | Manager-team-match assignments extracted from match records. |
| `artifacts/statsbomb_coverage.parquet` | Fetched season catalog entries, match counts, source URLs, update timestamps, and load status. |

The base artifacts are also available under `artifacts/`:

- `artifacts/match_observations.parquet`
- `artifacts/team_match_stats.parquet`
- `artifacts/data_dictionary.json`

## StatsBomb coverage

The ingestion module selects the male LaLiga entries in the StatsBomb
competition catalog from 2014/15 through 2020/21. The open-data selection is
not a complete league-season mirror; the release therefore records the actual
number of fetched matches instead of implying complete coverage.

| Season | Available matches |
| --- | ---: |
| 2014/15 | 38 |
| 2015/16 | 380 |
| 2016/17 | 34 |
| 2017/18 | 36 |
| 2018/19 | 34 |
| 2019/20 | 33 |
| 2020/21 | 35 |
| **Total** | **590** |

Use `statsbomb_coverage.parquet` as the authoritative release-level coverage
record. `minutes_played_estimate` is derived from lineup position intervals and
the observed event clock; it is an estimate, not an official minutes field.

## Leakage boundary

The pre-match table is generated after sorting matches chronologically by date,
kickoff time, and match ID. A match's result, goals, shots, corners, cards, and
other post-match statistics are added as labels or observations only after its
feature row has been created.

The StatsBomb lineups and events are observed-match artifacts. They are not
joined directly into `pre_match_forecasting`. A future availability or lineup
feature must use only information known before kickoff and must apply an
explicit timestamp cutoff. The final lineup, final event stream, or a
post-match player total cannot be used as a pre-match input.

Transfers, injuries, suspensions, player form, and as-of roster features are
not included in v0.2.0. Manager assignments in the enrichment bundle are
historical observations, not a current-manager feature table.

## Reference baseline

The included baseline is a standardized multinomial logistic regression over
the pre-match feature columns. The following results are a v0.2.0 reference
run, not a claim that the model generalizes beyond the held-out seasons.

| Split | Rows | Accuracy | Log loss | Multiclass Brier |
| --- | ---: | ---: | ---: | ---: |
| Validation (2024/25) | 380 | 0.5421 | 0.9861 | 0.5846 |
| Test (2025/26) | 380 | 0.5158 | 0.9807 | 0.5769 |

The metrics are intended as a reproducibility checkpoint. Future models should
also be compared with simple dynamic baselines, evaluated chronologically, and
calibrated before any practical use.

## Repository layout

```text
src/laliga_forecasting/build_dataset.py       # Base ingestion, features, outputs, Hub upload
src/laliga_forecasting/statsbomb_open_data.py # StatsBomb catalog, JSON ingestion, normalization
src/laliga_forecasting/train_baseline.py      # Chronological baseline evaluation
notebooks/01_build_dataset_colab.ipynb        # Colab build and publication workflow
requirements-colab.txt                         # Colab runtime dependencies
pyproject.toml                                 # Package metadata and local dependencies
```

Generated data, caches, model outputs, and credentials are excluded from the
Git repository. The build scripts retain source URLs and release metadata in
the generated artifacts.

## Reproduce locally

The local build downloads the base Football-Data.co.uk files and, when
requested, the StatsBomb JSON files. The StatsBomb raw responses are cached so
completed downloads can be reused on a later run.

```bash
git clone https://github.com/EF-Code/laliga-match-forecasting.git
cd laliga-match-forecasting

~/.venv/bin/python -m pip install -e .

PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.build_dataset \
  --output-dir outputs/v0.2.0 \
  --with-statsbomb

PYTHONPATH=src ~/.venv/bin/python -m laliga_forecasting.train_baseline \
  --input outputs/v0.2.0/pre_match_forecasting.parquet \
  --output outputs/v0.2.0/baseline_metrics.json
```

The local build never uploads data. The `--with-statsbomb` run requires network
access and processes a large event bundle; allow additional disk, memory, and
runtime for that step.

## Build from Google Colab

Open [`notebooks/01_build_dataset_colab.ipynb`](notebooks/01_build_dataset_colab.ipynb)
in Colab and choose **Runtime > Run all**. The notebook requires no Hugging Face
token, Google Drive mount, or manual input. It downloads the source data, builds
the dataset and StatsBomb enrichment, runs the baseline, and writes all outputs
to the temporary `/content/laliga-output` directory.

Colab storage is ephemeral. Download the generated files from the Colab file
browser if you want a local copy. The canonical published dataset is available
from the [Hugging Face dataset page](https://huggingface.co/datasets/NewSonnet/laliga-football-match-forecasting).

## Sources and attribution

The base results and basic match statistics come from
[Football-Data.co.uk Spain data](https://www.football-data.co.uk/spainm.php).

The enrichment comes from [StatsBomb Open Data](https://github.com/hudl/open-data).
Read the source repository's [README and terms](https://github.com/hudl/open-data/blob/master/README.md)
before reusing or redistributing source-derived artifacts. The README requests
StatsBomb attribution and logo use for published research, analysis, or
insights based on the open data; the Hub dataset card preserves that note.

The Hugging Face dataset is marked `license: other` because the upstream terms
and attribution requirements apply. Do not describe this release as an
unrestricted, open-licensed copy of the upstream data without reviewing those
terms.

## Roadmap

- Resolve team and player identities across source systems and seasons.
- Add transfer-window and registration history with effective dates.
- Add injury and suspension information with publication timestamps.
- Build as-of roster, manager, lineup-availability, and player-form features.
- Compare Elo, Poisson/Dixon–Coles, calibrated tree models, and probabilistic
  ensemble baselines.
- Add schema, provenance, leakage, and release validation checks to the build
  workflow.

## Contributing

Issues and pull requests are welcome. For data changes, include the source URL,
retrieval date, affected schema, coverage impact, and the time cutoff used for
any forecasting feature. Run the local build or a bounded parser check before
opening a pull request, and keep generated data and secrets out of commits.

## Licensing

This repository does not currently declare a separate code license. The
dataset and generated artifacts remain subject to the terms and attribution
requirements of their upstream sources; see the source links and the dataset
card before reuse.
