"""Build and optionally publish the first LaLiga forecasting dataset release.

The v0.1 release intentionally uses one consistent, freely downloadable
historical source: Football-Data.co.uk's Spain CSV files. It contains match
results and basic match statistics. Lineups, minutes, transfers, managers,
and availability are reserved for a later, separately documented enrichment
source so that the first release does not silently mix incompatible data.

All pre-match features are calculated in chronological order. Statistics from
the match being predicted never enter that match's feature row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_ROOT = "https://www.football-data.co.uk/mmz4281"
DATASET_SLUG = "laliga-football-match-forecasting"
DATASET_VERSION = "0.1.0"
SEASON_START_YEARS = tuple(range(2014, 2026))  # 2014-15 through 2025-26

REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
STAT_COLUMNS = (
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
)

TEAM_ALIASES = {
    "ath bilbao": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "ath madrid": "Atletico Madrid",
    "atletico madrid": "Atletico Madrid",
    "betis": "Real Betis",
    "celta": "Celta Vigo",
    "dep coruna": "Deportivo La Coruna",
    "deportivo la coruna": "Deportivo La Coruna",
    "la coruna": "Deportivo La Coruna",
    "espanol": "Espanyol",
    "espanyol": "Espanyol",
    "rayo vallecano": "Rayo Vallecano",
    "vallecano": "Rayo Vallecano",
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",
    "sp gijon": "Sporting Gijon",
    "sporting gijon": "Sporting Gijon",
}

FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_difference_before",
    "home_matches_seen",
    "away_matches_seen",
    "home_points_avg_last5",
    "away_points_avg_last5",
    "home_goals_for_avg_last5",
    "away_goals_for_avg_last5",
    "home_goals_against_avg_last5",
    "away_goals_against_avg_last5",
    "home_shots_for_avg_last5",
    "away_shots_for_avg_last5",
    "home_shots_against_avg_last5",
    "away_shots_against_avg_last5",
    "home_shots_on_target_for_avg_last5",
    "away_shots_on_target_for_avg_last5",
    "home_shots_on_target_against_avg_last5",
    "away_shots_on_target_against_avg_last5",
    "home_corners_for_avg_last5",
    "away_corners_for_avg_last5",
    "home_corners_against_avg_last5",
    "away_corners_against_avg_last5",
    "home_win_rate_last5",
    "away_win_rate_last5",
    "home_rest_days",
    "away_rest_days",
]


def _ascii_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def canonical_team(value: Any) -> str:
    key = _ascii_key(value)
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    return " ".join(part.capitalize() for part in key.split())


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def season_url(start_year: int) -> str:
    return f"{SOURCE_ROOT}/{season_code(start_year)}/SP1.csv"


def _to_number(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def fetch_season(start_year: int) -> pd.DataFrame:
    """Download and normalize one Football-Data.co.uk LaLiga season."""

    url = season_url(start_year)
    frame = pd.read_csv(url, encoding="latin-1")
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{url} is missing required columns: {missing}")

    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed", errors="coerce")
    frame["Time"] = frame.get("Time", pd.Series(index=frame.index, dtype="string")).astype("string")
    frame["HomeTeam"] = frame["HomeTeam"].map(canonical_team)
    frame["AwayTeam"] = frame["AwayTeam"].map(canonical_team)
    _to_number(frame, ["FTHG", "FTAG", "HTHG", "HTAG", *STAT_COLUMNS])

    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).copy()
    frame["season_start_year"] = start_year
    frame["season"] = season_label(start_year)
    frame["source_url"] = url
    frame["source_row"] = frame.index.astype(int)
    frame["match_id"] = [
        hashlib.sha1(
            f"{season_label(start_year)}|{date.date()}|{time}|{home}|{away}".encode("utf-8")
        ).hexdigest()[:16]
        for date, time, home, away in zip(
            frame["Date"], frame["Time"], frame["HomeTeam"], frame["AwayTeam"]
        )
    ]
    return frame.reset_index(drop=True)


def fetch_all_seasons(start_years: tuple[int, ...] = SEASON_START_YEARS) -> pd.DataFrame:
    frames = []
    failures: list[dict[str, Any]] = []
    for start_year in start_years:
        try:
            frame = fetch_season(start_year)
            frames.append(frame)
            print(f"LOADED {season_label(start_year)} rows={len(frame)}")
        except Exception as exc:  # keep the full run auditable and continue
            failures.append({"season": season_label(start_year), "url": season_url(start_year), "error": str(exc)})
            print(f"SKIPPED {season_label(start_year)} error={exc}")
    if not frames:
        raise RuntimeError("No LaLiga seasons could be loaded")
    if failures:
        print(f"SEASON_FAILURES {json.dumps(failures, sort_keys=True)}")
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Time", "match_id"]).reset_index(drop=True)


def _empty_history() -> dict[str, Any]:
    return {
        "date": None,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "shots_for": np.nan,
        "shots_against": np.nan,
        "shots_on_target_for": np.nan,
        "shots_on_target_against": np.nan,
        "corners_for": np.nan,
        "corners_against": np.nan,
        "points": 0.0,
        "win": 0.0,
    }


def _safe_mean(values: list[float], fallback: float) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else float(fallback)


def _team_snapshot(
    history: dict[str, deque[dict[str, Any]]],
    team: str,
    current_date: pd.Timestamp,
    priors: dict[str, float],
) -> dict[str, float]:
    rows = list(history.get(team, ()))
    recent = rows[-5:]
    last_date = rows[-1]["date"] if rows else None
    rest_days = (current_date - last_date).days if last_date is not None else 7

    def mean(key: str) -> float:
        return _safe_mean([row[key] for row in recent], priors[key])

    return {
        "matches_seen": float(len(rows)),
        "points_avg_last5": mean("points"),
        "goals_for_avg_last5": mean("goals_for"),
        "goals_against_avg_last5": mean("goals_against"),
        "shots_for_avg_last5": mean("shots_for"),
        "shots_against_avg_last5": mean("shots_against"),
        "shots_on_target_for_avg_last5": mean("shots_on_target_for"),
        "shots_on_target_against_avg_last5": mean("shots_on_target_against"),
        "corners_for_avg_last5": mean("corners_for"),
        "corners_against_avg_last5": mean("corners_against"),
        "win_rate_last5": mean("win"),
        "rest_days": float(max(0, min(rest_days, 60))),
    }


def _league_priors(frame: pd.DataFrame) -> dict[str, float]:
    pairs = {
        "goals_for": [frame["FTHG"], frame["FTAG"]],
        "goals_against": [frame["FTHG"], frame["FTAG"]],
        "shots_for": [frame.get("HS", pd.Series(dtype=float)), frame.get("AS", pd.Series(dtype=float))],
        "shots_against": [frame.get("HS", pd.Series(dtype=float)), frame.get("AS", pd.Series(dtype=float))],
        "shots_on_target_for": [frame.get("HST", pd.Series(dtype=float)), frame.get("AST", pd.Series(dtype=float))],
        "shots_on_target_against": [frame.get("HST", pd.Series(dtype=float)), frame.get("AST", pd.Series(dtype=float))],
        "corners_for": [frame.get("HC", pd.Series(dtype=float)), frame.get("AC", pd.Series(dtype=float))],
        "corners_against": [frame.get("HC", pd.Series(dtype=float)), frame.get("AC", pd.Series(dtype=float))],
        "points": [],
        "win": [],
    }
    priors: dict[str, float] = {}
    for key, series_list in pairs.items():
        if key == "points":
            priors[key] = 1.2
        elif key == "win":
            priors[key] = 1 / 3
        else:
            values = pd.concat(series_list, ignore_index=True) if series_list else pd.Series(dtype=float)
            priors[key] = _safe_mean(values.tolist(), 1.35 if "goals" in key else 5.0)
    return priors


def _points(result: str, home: bool) -> float:
    if result == "D":
        return 1.0
    if result == "H":
        return 3.0 if home else 0.0
    return 0.0 if home else 3.0


def _result_for(result: str, home: bool) -> float:
    return float((result == "H") if home else (result == "A"))


def build_pre_match_dataset(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe pre-match rows plus raw observation tables."""

    ordered = matches.sort_values(["Date", "Time", "match_id"]).reset_index(drop=True).copy()
    priors = _league_priors(ordered)
    history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))
    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    pre_match_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []

    for _, match in ordered.iterrows():
        date = pd.Timestamp(match["Date"])
        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        home_snapshot = _team_snapshot(history, home, date, priors)
        away_snapshot = _team_snapshot(history, away, date, priors)
        home_elo = float(elo[home])
        away_elo = float(elo[away])

        row: dict[str, Any] = {
            "match_id": str(match["match_id"]),
            "date": date.date().isoformat(),
            "season": str(match["season"]),
            "season_start_year": int(match["season_start_year"]),
            "home_team": home,
            "away_team": away,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "elo_difference_before": home_elo - away_elo,
            "home_matches_seen": int(home_snapshot["matches_seen"]),
            "away_matches_seen": int(away_snapshot["matches_seen"]),
            "home_points_avg_last5": home_snapshot["points_avg_last5"],
            "away_points_avg_last5": away_snapshot["points_avg_last5"],
            "home_goals_for_avg_last5": home_snapshot["goals_for_avg_last5"],
            "away_goals_for_avg_last5": away_snapshot["goals_for_avg_last5"],
            "home_goals_against_avg_last5": home_snapshot["goals_against_avg_last5"],
            "away_goals_against_avg_last5": away_snapshot["goals_against_avg_last5"],
            "home_shots_for_avg_last5": home_snapshot["shots_for_avg_last5"],
            "away_shots_for_avg_last5": away_snapshot["shots_for_avg_last5"],
            "home_shots_against_avg_last5": home_snapshot["shots_against_avg_last5"],
            "away_shots_against_avg_last5": away_snapshot["shots_against_avg_last5"],
            "home_shots_on_target_for_avg_last5": home_snapshot["shots_on_target_for_avg_last5"],
            "away_shots_on_target_for_avg_last5": away_snapshot["shots_on_target_for_avg_last5"],
            "home_shots_on_target_against_avg_last5": home_snapshot["shots_on_target_against_avg_last5"],
            "away_shots_on_target_against_avg_last5": away_snapshot["shots_on_target_against_avg_last5"],
            "home_corners_for_avg_last5": home_snapshot["corners_for_avg_last5"],
            "away_corners_for_avg_last5": away_snapshot["corners_for_avg_last5"],
            "home_corners_against_avg_last5": home_snapshot["corners_against_avg_last5"],
            "away_corners_against_avg_last5": away_snapshot["corners_against_avg_last5"],
            "home_win_rate_last5": home_snapshot["win_rate_last5"],
            "away_win_rate_last5": away_snapshot["win_rate_last5"],
            "home_rest_days": home_snapshot["rest_days"],
            "away_rest_days": away_snapshot["rest_days"],
            "home_goals": int(match["FTHG"]),
            "away_goals": int(match["FTAG"]),
            "result": str(match["FTR"]),
            "total_goals": int(match["FTHG"] + match["FTAG"]),
            "source_url": str(match["source_url"]),
            "dataset_version": DATASET_VERSION,
        }
        pre_match_rows.append(row)

        observation = {
            "match_id": str(match["match_id"]),
            "date": date.date().isoformat(),
            "season": str(match["season"]),
            "season_start_year": int(match["season_start_year"]),
            "home_team": home,
            "away_team": away,
            "home_goals": int(match["FTHG"]),
            "away_goals": int(match["FTAG"]),
            "result": str(match["FTR"]),
            "half_time_home_goals": match.get("HTHG"),
            "half_time_away_goals": match.get("HTAG"),
            "home_shots": match.get("HS"),
            "away_shots": match.get("AS"),
            "home_shots_on_target": match.get("HST"),
            "away_shots_on_target": match.get("AST"),
            "home_fouls": match.get("HF"),
            "away_fouls": match.get("AF"),
            "home_corners": match.get("HC"),
            "away_corners": match.get("AC"),
            "home_yellow_cards": match.get("HY"),
            "away_yellow_cards": match.get("AY"),
            "home_red_cards": match.get("HR"),
            "away_red_cards": match.get("AR"),
            "source_url": str(match["source_url"]),
            "source_row": int(match["source_row"]),
            "dataset_version": DATASET_VERSION,
        }
        for key, value in observation.items():
            if pd.isna(value):
                observation[key] = None
        observation_rows.append(observation)

        home_history = {
            "date": date,
            "goals_for": float(match["FTHG"]),
            "goals_against": float(match["FTAG"]),
            "shots_for": match.get("HS"),
            "shots_against": match.get("AS"),
            "shots_on_target_for": match.get("HST"),
            "shots_on_target_against": match.get("AST"),
            "corners_for": match.get("HC"),
            "corners_against": match.get("AC"),
            "points": _points(str(match["FTR"]), home=True),
            "win": _result_for(str(match["FTR"]), home=True),
        }
        away_history = {
            "date": date,
            "goals_for": float(match["FTAG"]),
            "goals_against": float(match["FTHG"]),
            "shots_for": match.get("AS"),
            "shots_against": match.get("HS"),
            "shots_on_target_for": match.get("AST"),
            "shots_on_target_against": match.get("HST"),
            "corners_for": match.get("AC"),
            "corners_against": match.get("HC"),
            "points": _points(str(match["FTR"]), home=False),
            "win": _result_for(str(match["FTR"]), home=False),
        }
        history[home].append(home_history)
        history[away].append(away_history)
        team_rows.extend(
            [
                {
                    "match_id": str(match["match_id"]),
                    "date": date.date().isoformat(),
                    "season": str(match["season"]),
                    "season_start_year": int(match["season_start_year"]),
                    "team": home,
                    "opponent": away,
                    "venue": "home",
                    "goals_for": int(match["FTHG"]),
                    "goals_against": int(match["FTAG"]),
                    "shots_for": match.get("HS"),
                    "shots_against": match.get("AS"),
                    "shots_on_target_for": match.get("HST"),
                    "shots_on_target_against": match.get("AST"),
                    "corners_for": match.get("HC"),
                    "corners_against": match.get("AC"),
                    "points": home_history["points"],
                },
                {
                    "match_id": str(match["match_id"]),
                    "date": date.date().isoformat(),
                    "season": str(match["season"]),
                    "season_start_year": int(match["season_start_year"]),
                    "team": away,
                    "opponent": home,
                    "venue": "away",
                    "goals_for": int(match["FTAG"]),
                    "goals_against": int(match["FTHG"]),
                    "shots_for": match.get("AS"),
                    "shots_against": match.get("HS"),
                    "shots_on_target_for": match.get("AST"),
                    "shots_on_target_against": match.get("HST"),
                    "corners_for": match.get("AC"),
                    "corners_against": match.get("HC"),
                    "points": away_history["points"],
                },
            ]
        )

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - 60) / 400))
        actual_home = 1.0 if match["FTR"] == "H" else 0.5 if match["FTR"] == "D" else 0.0
        k = 20.0
        elo[home] = home_elo + k * (actual_home - expected_home)
        elo[away] = away_elo - k * (actual_home - expected_home)

    pre_match = pd.DataFrame(pre_match_rows)
    observations = pd.DataFrame(observation_rows)
    team_stats = pd.DataFrame(team_rows)

    # Splits are chronological and season-based. The latest complete season is
    # reserved as a separate holdout for a future model report.
    def split_for(year: int) -> str:
        if year <= 2023:
            return "train"
        if year == 2024:
            return "validation"
        if year == 2025:
            return "test"
        return "unspecified"

    for frame in (pre_match, observations, team_stats):
        frame["split"] = frame["season_start_year"].map(split_for)

    return pre_match, observations, team_stats


def data_dictionary() -> dict[str, Any]:
    return {
        "dataset_name": "LaLiga Football Match Forecasting Dataset",
        "dataset_version": DATASET_VERSION,
        "source": "Football-Data.co.uk Spain SP1 historical CSV files",
        "source_url": "https://www.football-data.co.uk/spainm.php",
        "scope": "LaLiga match results and basic match statistics; no player, lineup, transfer, manager, or injury enrichment in v0.1.0.",
        "target": {
            "result": "H, D, or A full-time result",
            "home_goals": "Full-time home goals",
            "away_goals": "Full-time away goals",
        },
        "pre_match_rule": "All feature columns are calculated from matches earlier in chronological order. The current match's statistics are not used as features.",
        "splits": {
            "train": "2014-15 through 2023-24",
            "validation": "2024-25",
            "test": "2025-26",
        },
        "future_enrichment": ["lineups", "minutes", "transfers", "managers", "injuries", "player_form"],
        "license_note": "The source publisher's terms and attribution requirements apply. Verify redistribution rights before treating the Hub copy as an open-licensed dataset.",
        "feature_columns": FEATURE_COLUMNS,
    }


def write_outputs(
    output_dir: Path,
    pre_match: pd.DataFrame,
    observations: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_path = output_dir / "pre_match_forecasting.parquet"
    observations_path = output_dir / "match_observations.parquet"
    team_path = output_dir / "team_match_stats.parquet"
    dictionary_path = output_dir / "data_dictionary.json"
    pre_match.to_parquet(pre_path, index=False)
    observations.to_parquet(observations_path, index=False)
    team_stats.to_parquet(team_path, index=False)
    dictionary_path.write_text(json.dumps(data_dictionary(), indent=2) + "\n", encoding="utf-8")
    return {
        "pre_match": pre_path,
        "observations": observations_path,
        "team_stats": team_path,
        "dictionary": dictionary_path,
    }


def _colab_token() -> str:
    try:
        from google.colab import userdata  # type: ignore
    except ImportError as exc:
        raise RuntimeError("HF publication is only enabled from the Colab runtime") from exc
    token = userdata.get("HF_TOKEN")
    if not token or not isinstance(token, str):
        raise RuntimeError("Add a write-capable HF_TOKEN secret in Colab and enable notebook access")
    return token


def _hub_card(pre_match: pd.DataFrame, repo_id: str) -> str:
    split_counts = pre_match["split"].value_counts().sort_index().to_dict()
    return f"""---
pretty_name: LaLiga Football Match Forecasting Dataset
tags:
- football
- soccer
- laliga
- sports-analytics
- match-prediction
- time-series
- tabular
task_categories:
- tabular-classification
size_categories:
- 1K<n<10K
license: other
---

# LaLiga Football Match Forecasting Dataset

Version **{DATASET_VERSION}** is a reproducible, match-level starting point for
time-aware LaLiga forecasting. It contains historical full-time results and
basic match statistics from Football-Data.co.uk, plus features calculated only
from matches that occurred before each target match.

## Contents

- `pre_match_forecasting.parquet` — the main Hugging Face dataset with
  chronological `train`, `validation`, and `test` splits.
- `artifacts/match_observations.parquet` — observed match outcomes and
  post-match basic statistics. Do not use these columns as pre-match features.
- `artifacts/team_match_stats.parquet` — long-form observed team-match stats.
- `artifacts/data_dictionary.json` — field definitions and provenance notes.

Rows: **{len(pre_match)}**. Split counts: `{json.dumps(split_counts, sort_keys=True)}`.

## Important scope limitation

This release does **not** yet contain lineups, player minutes, transfers,
managers, injuries, or player-level form. Those will be added as a separately
documented enrichment release after source coverage and redistribution terms
are verified. Do not describe this release as player-aware.

## Leakage boundary

The `pre_match_forecasting` features are generated sequentially. The target
match's goals, shots, corners, cards, and other post-match statistics are not
used in its feature row. Use season-based splits and do not randomly shuffle
rows when evaluating a forecaster.

## Source and attribution

Source: [Football-Data.co.uk Spain data](https://www.football-data.co.uk/spainm.php).
The source publisher's terms and attribution requirements apply. Verify
redistribution rights before using this Hub copy in a public product.

Repository: https://github.com/EF-Code/laliga-match-forecasting
Dataset repository: https://huggingface.co/datasets/{repo_id}
"""


def publish_to_hub(
    output_dir: Path,
    pre_match: pd.DataFrame,
    paths: dict[str, Path],
    token: str | None = None,
) -> str:
    """Publish generated data with an explicitly supplied or Colab secret token.

    The notebook supplies the token after reading it in the Colab kernel. The
    fallback keeps direct notebook use possible, while avoiding prompts and
    never printing the token.
    """

    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install requirements-colab.txt before publishing") from exc

    if token is None:
        token = _colab_token()
    if not token or not isinstance(token, str):
        raise RuntimeError("Add a write-capable HF_TOKEN secret in Colab and enable notebook access")
    api = HfApi(token=token)
    account = api.whoami()
    username = account.get("name") or account.get("auth", {}).get("name")
    if not username:
        raise RuntimeError("Could not resolve the Hugging Face account from the secret")
    repo_id = f"{username}/{DATASET_SLUG}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)

    split_frames = {
        split: pre_match.loc[pre_match["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    dataset_dict = DatasetDict(
        {split: Dataset.from_pandas(frame, preserve_index=False) for split, frame in split_frames.items()}
    )
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=False,
        max_shard_size="100MB",
        commit_message=f"Publish LaLiga forecasting dataset v{DATASET_VERSION}",
    )

    for key, path in (
        ("match_observations", paths["observations"]),
        ("team_match_stats", paths["team_stats"]),
        ("data_dictionary", paths["dictionary"]),
    ):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"artifacts/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Add {key} artifact for v{DATASET_VERSION}",
        )

    card_path = output_dir / "README.md"
    card_path.write_text(_hub_card(pre_match, repo_id), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card and provenance",
    )
    print(f"HF_DATASET_REPO https://huggingface.co/datasets/{repo_id}")
    print(f"HF_DATASET_ROWS {len(pre_match)}")
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v0.1.0"))
    parser.add_argument("--publish", action="store_true", help="Publish to the HF account from Colab HF_TOKEN")
    args = parser.parse_args()

    matches = fetch_all_seasons()
    pre_match, observations, team_stats = build_pre_match_dataset(matches)
    paths = write_outputs(args.output_dir, pre_match, observations, team_stats)
    print(f"MATCH_ROWS {len(matches)}")
    print(f"PRE_MATCH_ROWS {len(pre_match)}")
    print(f"TEAM_MATCH_ROWS {len(team_stats)}")
    print(f"SPLITS {json.dumps(pre_match['split'].value_counts().sort_index().to_dict(), sort_keys=True)}")
    print(f"OUTPUT_DIR {args.output_dir.resolve()}")

    if args.publish:
        publish_to_hub(args.output_dir, pre_match, paths)


if __name__ == "__main__":
    main()
