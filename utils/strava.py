import logging
import os
import time
import json
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from urllib.parse import urlencode
import config

log = logging.getLogger(__name__)

os.makedirs(config.CACHE_DIR, exist_ok=True)

def _save_tokens(tokens: Dict[str, Any]):
    with open(config.TOKENS_PATH, "w") as f:
        json.dump(tokens, f)

def _load_tokens() -> Dict[str, Any] | None:
    if not os.path.exists(config.TOKENS_PATH):
        return None
    with open(config.TOKENS_PATH, "r") as f:
        return json.load(f)

def build_auth_url(client_id: str, redirect_uri: str, scope: str="read,activity:read_all", approval_prompt: str="auto") -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": approval_prompt,
        "scope": scope,
    }
    return f"{config.STRAVA_AUTH_URL}?{urlencode(params)}"

def exchange_code_for_token(client_id: str, client_secret: str, code: str) -> Dict[str, Any]:
    payload = {"client_id": client_id, "client_secret": client_secret, "code": code, "grant_type": "authorization_code"}
    r = requests.post(config.STRAVA_TOKEN_URL, data=payload, timeout=config.DEFAULT_TIMEOUT)
    r.raise_for_status()
    tokens = r.json()
    tokens["obtained_at"] = int(time.time())
    tokens["client_id_used"] = str(client_id)
    _save_tokens(tokens)
    return tokens

def _refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> Dict[str, Any]:
    payload = {"client_id": client_id, "client_secret": client_secret, "grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(config.STRAVA_TOKEN_URL, data=payload, timeout=config.DEFAULT_TIMEOUT); r.raise_for_status()
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} {r.text}", response=r)
    tokens = r.json(); tokens["obtained_at"] = int(time.time()); _save_tokens(tokens);
    return tokens

def _get_headers(access_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}

def _disconnect_strava():
    try:
        os.remove(config.TOKENS_PATH)
    except FileNotFoundError:
        pass

def ensure_token(client_id: str, client_secret: str) -> Dict[str, Any] | None:
    tokens = _load_tokens()
    if not tokens:
        return None
    # If current client_id doesn't match the one that created the tokens, force reconnect
    if str(tokens.get("client_id_used","")) != str(client_id):
        return None
    try:
        expires_at = tokens.get("expires_at", 0)
        if int(time.time()) >= int(expires_at) - 60:
            tokens = _refresh_access_token(client_id, client_secret, tokens["refresh_token"])
            tokens["client_id_used"] = str(client_id)
            _save_tokens(tokens)
        return tokens
    except requests.HTTPError as e:
        # If refresh fails (400/401), nuke tokens so UI shows "Connect Strava" again
        if getattr(e, "response", None) is not None and e.response.status_code in (400, 401):
            _disconnect_strava()
            return None
        raise

def list_activities(access_token: str, per_page: int=200) -> List[Dict[str, Any]]:
    activities = []; headers = _get_headers(access_token); page = 1
    params = {"per_page": min(per_page, 200)}
    while True:
        params["page"] = page
        r = requests.get(f"{config.API_BASE}/athlete/activities", headers=headers, params=params, timeout=config.DEFAULT_TIMEOUT)
        if r.status_code == 429: raise RuntimeError("Rate limit hit. Try again later.")
        r.raise_for_status(); batch = r.json()
        if not batch: break
        activities.extend(batch)
        if len(batch) < params["per_page"]: break
        page += 1
        if page > 10: break

    # De-duplicate by activity id (safety)
    seen = set()
    dedup = []
    for a in activities:
        aid = a.get("id")
        if aid in seen:
            continue
        seen.add(aid)
        dedup.append(a)
    return dedup

def is_run(a: Dict[str, Any]) -> bool:
    stype = a.get("sport_type") or a.get("type")
    return stype in ("Run","TrailRun","VirtualRun")

def is_race(a: Dict[str, Any]) -> bool:
    # Run Type 'Race' appears as workout_type == 1
    return a.get("workout_type") == 1


def is_hard_effort(a: Dict[str, Any], hr_threshold: int = 150) -> bool:
    """True if activity is a long run (2), workout (3), has avg HR above threshold, or suffer score > 50."""
    if a.get("workout_type") in (2, 3):
        return True
    if (a.get("average_heartrate") or 0) > hr_threshold:
        return True
    if (a.get("suffer_score") or 0) > 130:
        return True
    return False


def compute_hr_threshold(
        activities: List[Dict[str, Any]],
        fallback: int = 150,
        min_activities: int = 10,
        percentile: float = 70.0,
) -> tuple[int, int]:
    """Return (threshold_bpm, n_runs) for hard-effort HR detection.

    Computes the given percentile of average_heartrate across all runs
    from the last 3 months. Returns (fallback, n_runs) when fewer than
    `min_activities` runs have HR data.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=91)
    hr_values = []
    for a in activities:
        if not is_run(a):
            continue
        try:
            dt = datetime.fromisoformat(a.get("start_date", "").replace("Z", "+00:00"))
            if dt < cutoff:
                continue
        except (ValueError, AttributeError):
            continue
        hr = a.get("average_heartrate")
        if hr and hr > 0:
            hr_values.append(float(hr))

    n_runs = len(hr_values)

    if n_runs < min_activities:
        log.info(
            "HR threshold: only %d run(s) with HR data in last 3 months (need %d) — using fallback %d bpm",
            n_runs, min_activities, fallback,
        )
        return fallback, n_runs

    threshold = int(np.percentile(hr_values, percentile))
    log.info(
        "HR threshold: %.0fth percentile of %d runs = %d bpm",
        percentile, n_runs, threshold,
    )
    return threshold, n_runs

def get_activity_streams(access_token: str, activity_id: int, types=None) -> Dict[str, Any]:
    if types is None:
        types = ["time","distance","altitude","velocity_smooth","grade_smooth","moving","latlng"]

    # --- local cache first ---
    path = os.path.join(config.CACHE_DIR, f"streams_{activity_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    # --- call Strava ---
    r = requests.get(
        f"{config.API_BASE}/activities/{activity_id}/streams",
        headers=_get_headers(access_token),
        params={"keys": ",".join(types), "key_by_type":"true"},
        timeout=config.DEFAULT_TIMEOUT
    )

    # Rate limit → show friendly wait time
    if r.status_code == 429:
        reset = int(r.headers.get("X-RateLimit-Reset", "0"))
        wait = max(0, reset - int(time.time()))
        raise RuntimeError(f"Strava rate limit hit. Try again in ~{wait} seconds.")

    # 404 happens (manual/treadmill/no streams/odd privacy). Cache empty and skip next time.
    if r.status_code == 404:
        data = {}
        with open(path, "w") as f:
            json.dump(data, f)
        return data

    r.raise_for_status()
    data = r.json()

    # Save to cache so we never fetch this activity again
    with open(path, "w") as f:
        json.dump(data, f)

    return data
