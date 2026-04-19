"""
Data persistence functions for saving and loading models/credentials
"""

import os
import json
import pandas as pd
from pathlib import Path
import streamlit as st
import config
from dotenv import load_dotenv

load_dotenv()

def _athlete_dir(athlete_id) -> str:
    """Return the data directory for a specific athlete."""
    if athlete_id:
        return os.path.join(config.DATA_DIR, f"athlete_{athlete_id}")
    return config.DATA_DIR


def save_pace_model_to_disk(pace_model, athlete_id=None):
    """
    Persist pace model to CSV/JSON files, scoped to the given athlete.

    Args:
        pace_model: PaceModel object to save
        athlete_id: Strava athlete ID string (None falls back to shared dir)
    """
    try:
        d = _athlete_dir(athlete_id)
        Path(d).mkdir(parents=True, exist_ok=True)
        pace_model.pace_df.to_csv(os.path.join(d, "pace_curves.csv"), index=False)
        pace_model.used_races.to_csv(os.path.join(d, "used_races.csv"), index=False)
        with open(os.path.join(d, "model_meta.json"), "w") as f:
            json.dump(pace_model.meta, f)
    except Exception as e:
        st.warning(f"Could not save model to disk: {e}")


def load_pace_model_from_disk(athlete_id=None):
    """
    Load pace model from disk if available, scoped to the given athlete.

    Returns:
        PaceModel object if successfully loaded, None otherwise
    """
    d = _athlete_dir(athlete_id)
    curves = os.path.join(d, "pace_curves.csv")
    races = os.path.join(d, "used_races.csv")
    meta_path = os.path.join(d, "model_meta.json")

    if not all([os.path.exists(curves), os.path.exists(races), os.path.exists(meta_path)]):
        return None

    try:
        from models import PaceModel
        pace_df = pd.read_csv(curves)
        used_races = pd.read_csv(races)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return PaceModel(pace_df, used_races, meta)
    except Exception as e:
        print(f"Error loading pace model from disk: {e}")
        return None

def get_app_creds() -> dict:
    """
    Return Strava app credentials without ever exposing them in the UI.
    Reads from st.secrets first (Streamlit Cloud / secrets.toml),
    then falls back to environment variables / .env file.
    """
    try:
        client_id = st.secrets["STRAVA_CLIENT_ID"]
        client_secret = st.secrets["STRAVA_CLIENT_SECRET"]
        return {"client_id": str(client_id), "client_secret": str(client_secret)}
    except KeyError:
        pass  # keys not present in st.secrets — try env vars
    except Exception:
        pass  # st.secrets unavailable (no secrets.toml locally) — try env vars
    return {
        "client_id": os.environ.get("STRAVA_CLIENT_ID", ""),
        "client_secret": os.environ.get("STRAVA_CLIENT_SECRET", ""),
    }


def load_saved_app_creds(app_credits_path: str):
    try:
        with open(app_credits_path, "r") as f:
            data = json.load(f)
        if data.get("client_id") and data.get("client_secret"):
            return data
    except Exception:
        pass
    return get_app_creds()

def save_app_creds(client_id: str, client_secret: str, data_dir: str, app_credits_path: str):
    os.makedirs(data_dir, exist_ok=True)
    with open(app_credits_path, "w") as f:
        json.dump({"client_id": str(client_id), "client_secret": str(client_secret)}, f)

def forget_app_creds(app_credits_path: str):
    try:
        os.remove(app_credits_path)
    except FileNotFoundError:
        pass


def load_excluded_race_ids(athlete_id=None) -> set:
    """Load set of race IDs the user has chosen to exclude, scoped to athlete."""
    d = _athlete_dir(athlete_id)
    path = os.path.join(d, "excluded_races.csv")
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["id"].astype(str))
    except Exception:
        return set()


def save_excluded_race_ids(excluded_ids: set, athlete_id=None):
    """Persist the set of excluded race IDs to CSV, scoped to athlete."""
    d = _athlete_dir(athlete_id)
    Path(d).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": sorted(excluded_ids)})
    df.to_csv(os.path.join(d, "excluded_races.csv"), index=False)


def load_streams(race_id) -> dict | None:
    """Load cached streams JSON. Returns None if missing or empty."""
    path = os.path.join(config.CACHE_DIR, f"streams_{race_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None