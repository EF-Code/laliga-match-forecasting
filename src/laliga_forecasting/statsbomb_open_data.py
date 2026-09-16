"""Download and normalize the selected StatsBomb Open Data LaLiga history.

StatsBomb publishes selected competitions and seasons as JSON for public
research and football analytics.  This module keeps that enrichment separate
from the results-derived pre-match feature table: the lineups and events are
observed match artifacts, not automatically valid pre-match features.

The raw JSON is cached locally so a rerun does not redownload completed files.
The generated tables retain source IDs, URLs, and serialized nested payloads
where a flat schema would otherwise discard information.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

STATSBOMB_DATA_ROOT = "https://raw.githubusercontent.com/hudl/open-data/master/data"
STATSBOMB_REPOSITORY_URL = "https://github.com/hudl/open-data"
STATSBOMB_TERMS_URL = "https://github.com/hudl/open-data/blob/master/README.md"
STATSBOMB_COMPETITION_ID = 11
STATSBOMB_COMPETITION_NAME = "La Liga"
DEFAULT_MIN_SEASON_START_YEAR = 2014
DEFAULT_MAX_SEASON_START_YEAR = 2020


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _person_fields(person: dict[str, Any] | None) -> dict[str, Any]:
    person = person or {}
    country = person.get("country") or {}
    return {
        "id": person.get("id"),
        "name": person.get("name"),
        "nickname": person.get("nickname"),
        "dob": person.get("dob"),
        "country_id": country.get("id"),
        "country_name": country.get("name"),
    }


def _season_start_year(season_name: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(season_name))
    return int(match.group(0)) if match else None


def _clock_seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parts = [float(part) for part in str(value).split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _event_clock_seconds(event: dict[str, Any]) -> float:
    minute = _number(event.get("minute")) or 0
    second = _number(event.get("second")) or 0
    return float(minute * 60 + second)


def _match_duration_seconds(events: list[dict[str, Any]]) -> float:
    """Estimate a match clock for players whose final position ends at None."""

    if not events:
        return 90 * 60.0
    return max(90 * 60.0, max(_event_clock_seconds(event) for event in events))


def _get_json(
    session: requests.Session,
    url: str,
    cache_path: Path | None = None,
    timeout_seconds: int = 90,
) -> Any:
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"temporary HTTP {response.status_code} for {url}")
            response.raise_for_status()
            payload = response.json()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(_json_text(payload), encoding="utf-8")
            return payload
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download StatsBomb JSON: {url}: {last_error}")


def fetch_season_catalog(
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the available male La Liga seasons in the Open Data catalog."""

    owns_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "laliga-match-forecasting/0.2.0 research dataset"})
    cache_path = cache_dir / "competitions.json" if cache_dir else None
    payload = _get_json(session, f"{STATSBOMB_DATA_ROOT}/competitions.json", cache_path)
    rows = []
    for competition in payload:
        if competition.get("competition_id") != STATSBOMB_COMPETITION_ID:
            continue
        if competition.get("competition_gender") != "male":
            continue
        start_year = _season_start_year(competition.get("season_name"))
        if start_year is None:
            continue
        rows.append(
            {
                "competition_id": int(competition["competition_id"]),
                "competition_name": str(competition.get("competition_name") or STATSBOMB_COMPETITION_NAME),
                "season_id": int(competition["season_id"]),
                "season": str(competition.get("season_name")),
                "season_start_year": start_year,
                "match_available": competition.get("match_available"),
                "match_updated": competition.get("match_updated"),
                "source_url": f"{STATSBOMB_DATA_ROOT}/matches/{STATSBOMB_COMPETITION_ID}/{int(competition['season_id'])}.json",
            }
        )
    if owns_session:
        session.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("StatsBomb Open Data has no male La Liga seasons in competitions.json")
    return frame.sort_values(["season_start_year", "season_id"]).reset_index(drop=True)


def _normalize_match(
    match: dict[str, Any],
    season: dict[str, Any],
) -> dict[str, Any]:
    home = match.get("home_team") or {}
    away = match.get("away_team") or {}
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    home_managers = home.get("managers") or []
    away_managers = away.get("managers") or []

    def score_value(score: Any, key: str = "display") -> int | float | None:
        if isinstance(score, dict):
            return _number(score.get(key))
        return _number(score) if key == "display" else None

    return {
        "source": "statsbomb_open_data",
        "competition_id": int(season["competition_id"]),
        "competition_name": str(season["competition_name"]),
        "season_id": int(season["season_id"]),
        "season": str(season["season"]),
        "season_start_year": int(season["season_start_year"]),
        "statsbomb_match_id": int(match["match_id"]),
        "match_date": match.get("match_date"),
        "kick_off": match.get("kick_off"),
        "match_week": _number(match.get("match_week")),
        "match_status": match.get("match_status"),
        "competition_stage": (match.get("competition_stage") or {}).get("name"),
        "home_team_id": home.get("home_team_id"),
        "home_team_name": home.get("home_team_name"),
        "away_team_id": away.get("away_team_id"),
        "away_team_name": away.get("away_team_name"),
        "home_score": score_value(home_score),
        "away_score": score_value(away_score),
        "home_score_period1": score_value(home_score, "period1"),
        "away_score_period1": score_value(away_score, "period1"),
        "home_manager_ids": _json_text([manager.get("id") for manager in home_managers]),
        "away_manager_ids": _json_text([manager.get("id") for manager in away_managers]),
        "home_managers_json": _json_text(home_managers),
        "away_managers_json": _json_text(away_managers),
        "stadium_json": _json_text(match.get("stadium") or {}),
        "metadata_json": _json_text(match.get("metadata") or {}),
        "source_url": season["source_url"],
        "source_repository": STATSBOMB_REPOSITORY_URL,
        "source_terms_url": STATSBOMB_TERMS_URL,
    }


def _normalize_manager_rows(match: dict[str, Any], season: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for venue, team_key in (
        ("home", "home_team"),
        ("away", "away_team"),
    ):
        team = match.get(team_key) or {}
        for manager in team.get("managers") or []:
            person = _person_fields(manager)
            rows.append(
                {
                    "source": "statsbomb_open_data",
                    "competition_id": int(season["competition_id"]),
                    "season_id": int(season["season_id"]),
                    "season": str(season["season"]),
                    "season_start_year": int(season["season_start_year"]),
                    "statsbomb_match_id": int(match["match_id"]),
                    "team_id": team.get(f"{venue}_team_id"),
                    "team_name": team.get(f"{venue}_team_name"),
                    "venue": venue,
                    "manager_id": person["id"],
                    "manager_name": person["name"],
                    "manager_nickname": person["nickname"],
                    "manager_dob": person["dob"],
                    "manager_country_id": person["country_id"],
                    "manager_country_name": person["country_name"],
                    "source_url": season["source_url"],
                    "source_repository": STATSBOMB_REPOSITORY_URL,
                    "source_terms_url": STATSBOMB_TERMS_URL,
                }
            )
    return rows


def _normalize_lineup_rows(
    match: dict[str, Any],
    season: dict[str, Any],
    lineup_payload: list[dict[str, Any]],
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for team_block in lineup_payload:
        team_id = team_block.get("team_id")
        team_name = team_block.get("team_name")
        for player in team_block.get("lineup") or []:
            positions = player.get("positions") or []
            cards = player.get("cards") or []
            person_country = player.get("country") or {}
            start_reasons = [str(position.get("start_reason") or "") for position in positions]
            is_starter = any("Starting XI" in reason for reason in start_reasons)
            appeared = bool(positions)
            from_seconds = [value for value in (_clock_seconds(p.get("from")) for p in positions) if value is not None]
            to_seconds = [value for value in (_clock_seconds(p.get("to")) for p in positions) if value is not None]
            playing_seconds = 0.0
            for position in positions:
                start = _clock_seconds(position.get("from"))
                end = _clock_seconds(position.get("to"))
                if start is not None:
                    playing_seconds += max(0.0, (end if end is not None else duration_seconds) - start)

            player_id = player.get("player_id")
            player_rows.append(
                {
                    "source": "statsbomb_open_data",
                    "player_id": player_id,
                    "player_name": player.get("player_name"),
                    "player_nickname": player.get("player_nickname"),
                    "country_id": person_country.get("id"),
                    "country_name": person_country.get("name"),
                    "source_repository": STATSBOMB_REPOSITORY_URL,
                    "source_terms_url": STATSBOMB_TERMS_URL,
                }
            )
            lineup_rows.append(
                {
                    "source": "statsbomb_open_data",
                    "competition_id": int(season["competition_id"]),
                    "season_id": int(season["season_id"]),
                    "season": str(season["season"]),
                    "season_start_year": int(season["season_start_year"]),
                    "statsbomb_match_id": int(match["match_id"]),
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player.get("player_name"),
                    "player_nickname": player.get("player_nickname"),
                    "jersey_number": player.get("jersey_number"),
                    "country_id": person_country.get("id"),
                    "country_name": person_country.get("name"),
                    "is_starter": bool(is_starter),
                    "was_substitute": bool(not is_starter),
                    "appeared": bool(appeared),
                    "first_position": (positions[0] or {}).get("position") if positions else None,
                    "from_seconds": min(from_seconds) if from_seconds else None,
                    "to_seconds": max(to_seconds) if to_seconds else (duration_seconds if appeared else None),
                    "minutes_played_estimate": round(playing_seconds / 60.0, 3) if appeared else 0.0,
                    "positions_json": _json_text(positions),
                    "cards_json": _json_text(cards),
                    "source_url": f"{STATSBOMB_DATA_ROOT}/lineups/{int(match['match_id'])}.json",
                    "source_repository": STATSBOMB_REPOSITORY_URL,
                    "source_terms_url": STATSBOMB_TERMS_URL,
                }
            )
    return lineup_rows, player_rows


def _normalize_event_rows(
    match: dict[str, Any],
    season: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type") or {}
        team = event.get("team") or {}
        possession_team = event.get("possession_team") or {}
        player = event.get("player") or {}
        substitution = event.get("substitution") or {}
        replacement = substitution.get("replacement") or {}
        recipient = (event.get("pass") or {}).get("recipient") or {}
        location = event.get("location") or []
        end_location = (
            (event.get("pass") or {}).get("end_location")
            or (event.get("carry") or {}).get("end_location")
            or (event.get("shot") or {}).get("end_location")
            or []
        )
        rows.append(
            {
                "source": "statsbomb_open_data",
                "competition_id": int(season["competition_id"]),
                "season_id": int(season["season_id"]),
                "season": str(season["season"]),
                "season_start_year": int(season["season_start_year"]),
                "statsbomb_match_id": int(match["match_id"]),
                "event_id": event.get("id"),
                "event_index": _number(event.get("index")),
                "period": _number(event.get("period")),
                "timestamp": event.get("timestamp"),
                "minute": _number(event.get("minute")),
                "second": _number(event.get("second")),
                "duration": _number(event.get("duration")),
                "event_type_id": event_type.get("id"),
                "event_type_name": event_type.get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "possession_team_id": possession_team.get("id"),
                "possession_team_name": possession_team.get("name"),
                "play_pattern_id": (event.get("play_pattern") or {}).get("id"),
                "play_pattern_name": (event.get("play_pattern") or {}).get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "related_player_id": replacement.get("id") or recipient.get("id"),
                "related_player_name": replacement.get("name") or recipient.get("name"),
                "location_x": _number(location[0]) if len(location) > 0 else None,
                "location_y": _number(location[1]) if len(location) > 1 else None,
                "end_location_x": _number(end_location[0]) if len(end_location) > 0 else None,
                "end_location_y": _number(end_location[1]) if len(end_location) > 1 else None,
                "event_json": _json_text(event),
                "source_url": f"{STATSBOMB_DATA_ROOT}/events/{int(match['match_id'])}.json",
                "source_repository": STATSBOMB_REPOSITORY_URL,
                "source_terms_url": STATSBOMB_TERMS_URL,
            }
        )
    return rows


def build_statsbomb_bundle(
    cache_dir: Path,
    min_season_start_year: int = DEFAULT_MIN_SEASON_START_YEAR,
    max_season_start_year: int = DEFAULT_MAX_SEASON_START_YEAR,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Build normalized StatsBomb match, lineup, event, player, and manager tables."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    owns_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "laliga-match-forecasting/0.2.0 research dataset"})

    catalog = fetch_season_catalog(session=session, cache_dir=cache_dir)
    catalog = catalog.loc[
        catalog["season_start_year"].between(min_season_start_year, max_season_start_year)
    ].copy()
    if catalog.empty:
        raise ValueError("No StatsBomb La Liga seasons match the requested year range")

    match_rows: list[dict[str, Any]] = []
    raw_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for season in catalog.to_dict("records"):
        season_id = int(season["season_id"])
        try:
            season_matches = _get_json(
                session,
                season["source_url"],
                cache_dir / "matches" / f"{season_id}.json",
            )
            season_matches = season_matches if isinstance(season_matches, list) else []
            for match in season_matches:
                match_rows.append(_normalize_match(match, season))
                raw_matches.append((match, season))
            coverage_rows.append(
                {
                    "season": season["season"],
                    "season_id": season_id,
                    "season_start_year": season["season_start_year"],
                    "matches_available": len(season_matches),
                    "match_source_url": season["source_url"],
                    "match_updated": season["match_updated"],
                    "status": "loaded",
                }
            )
            print(f"STATSBOMB_SEASON {season['season']} matches={len(season_matches)}")
        except Exception as exc:
            failures.append({"season": season["season"], "season_id": season_id, "kind": "matches", "error": str(exc)})
            coverage_rows.append(
                {
                    "season": season["season"],
                    "season_id": season_id,
                    "season_start_year": season["season_start_year"],
                    "matches_available": 0,
                    "match_source_url": season["source_url"],
                    "match_updated": season["match_updated"],
                    "status": "failed",
                }
            )
            print(f"STATSBOMB_SEASON_FAILURE {season['season']} error={exc}")

    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    manager_rows: list[dict[str, Any]] = []
    for index, (match, season) in enumerate(raw_matches, start=1):
        match_id = int(match["match_id"])
        try:
            events = _get_json(
                session,
                f"{STATSBOMB_DATA_ROOT}/events/{match_id}.json",
                cache_dir / "events" / f"{match_id}.json",
            )
            events = events if isinstance(events, list) else []
            event_rows.extend(_normalize_event_rows(match, season, events))
        except Exception as exc:
            failures.append({"season": season["season"], "match_id": match_id, "kind": "events", "error": str(exc)})
            events = []

        try:
            lineup_payload = _get_json(
                session,
                f"{STATSBOMB_DATA_ROOT}/lineups/{match_id}.json",
                cache_dir / "lineups" / f"{match_id}.json",
            )
            lineup_payload = lineup_payload if isinstance(lineup_payload, list) else []
            normalized_lineups, normalized_players = _normalize_lineup_rows(
                match,
                season,
                lineup_payload,
                _match_duration_seconds(events),
            )
            lineup_rows.extend(normalized_lineups)
            player_rows.extend(normalized_players)
        except Exception as exc:
            failures.append({"season": season["season"], "match_id": match_id, "kind": "lineups", "error": str(exc)})

        normalized_managers = _normalize_manager_rows(match, season)
        manager_rows.extend(normalized_managers)
        if index == 1 or index % 50 == 0 or index == len(raw_matches):
            print(
                f"STATSBOMB_DETAIL {index}/{len(raw_matches)} "
                f"lineups={len(lineup_rows)} events={len(event_rows)} players={len(player_rows)}"
            )

    def sorted_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        return frame.sort_values(columns).reset_index(drop=True) if not frame.empty else frame

    matches_frame = sorted_frame(match_rows, ["match_date", "kick_off", "statsbomb_match_id"])
    lineups_frame = sorted_frame(lineup_rows, ["season_start_year", "statsbomb_match_id", "team_id", "player_id"])
    events_frame = sorted_frame(event_rows, ["season_start_year", "statsbomb_match_id", "event_index"])
    managers_frame = sorted_frame(manager_rows, ["season_start_year", "statsbomb_match_id", "team_id", "manager_id"])
    players_frame = pd.DataFrame(player_rows)
    if not players_frame.empty:
        players_frame = players_frame.drop_duplicates(subset=["player_id"], keep="first").sort_values(["player_name", "player_id"]).reset_index(drop=True)
    coverage_frame = pd.DataFrame(coverage_rows).sort_values(["season_start_year", "season_id"]).reset_index(drop=True)
    failure_frame = pd.DataFrame(failures)

    if not failure_frame.empty:
        print(f"STATSBOMB_FAILURES {len(failure_frame)}")
        failure_frame.to_json(cache_dir / "failures.json", orient="records", indent=2, force_ascii=False)

    if owns_session:
        session.close()

    return {
        "statsbomb_matches": matches_frame,
        "statsbomb_lineups": lineups_frame,
        "statsbomb_events": events_frame,
        "statsbomb_players": players_frame,
        "statsbomb_managers": managers_frame,
        "statsbomb_coverage": coverage_frame,
    }


def statsbomb_data_dictionary() -> dict[str, Any]:
    return {
        "source": "StatsBomb Open Data",
        "source_url": STATSBOMB_REPOSITORY_URL,
        "terms_url": STATSBOMB_TERMS_URL,
        "competition": "La Liga, male",
        "coverage_rule": (
            f"All male La Liga seasons in competitions.json from {DEFAULT_MIN_SEASON_START_YEAR}/{DEFAULT_MIN_SEASON_START_YEAR + 1} "
            f"through {DEFAULT_MAX_SEASON_START_YEAR}/{DEFAULT_MAX_SEASON_START_YEAR + 1}; the coverage artifact records the actual match count "
            "because the Open Data selection is not guaranteed to be a complete league-season mirror."
        ),
        "publication_note": (
            "The StatsBomb Open Data README describes selected data as freely available for public research and football analytics and asks "
            "published research, analysis, or insights to credit StatsBomb and use its logo. Preserve those attribution requirements and "
            "review the current terms before redistributing raw source files."
        ),
        "tables": {
            "statsbomb_matches": "One row per available StatsBomb La Liga match with scores, teams, season, and manager JSON references.",
            "statsbomb_lineups": "One row per listed player and team per match, including starter/appearance flags, position intervals, cards, and an estimated played-minute field.",
            "statsbomb_events": "One row per StatsBomb event with common time/team/player fields and the complete nested event serialized as event_json.",
            "statsbomb_players": "Deduplicated player dimension keyed by StatsBomb player_id from the lineup history.",
            "statsbomb_managers": "One row per manager-team-match assignment from the match records.",
            "statsbomb_coverage": "Actual season match counts, source URLs, update timestamps, and load status.",
        },
        "leakage_boundary": "These are observed-match artifacts. They are not joined into pre_match_forecasting until a feature is constructed using only information available before kickoff.",
    }


__all__ = [
    "DEFAULT_MAX_SEASON_START_YEAR",
    "DEFAULT_MIN_SEASON_START_YEAR",
    "STATSBOMB_COMPETITION_ID",
    "STATSBOMB_DATA_ROOT",
    "STATSBOMB_REPOSITORY_URL",
    "STATSBOMB_TERMS_URL",
    "build_statsbomb_bundle",
    "fetch_season_catalog",
    "statsbomb_data_dictionary",
]
