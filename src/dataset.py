from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

from src.config import RAW_DIR

DIVVY_BUCKET_URL = "https://divvy-tripdata.s3.amazonaws.com/"


def generate_synthetic_divvy_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="h")
    base = pd.DataFrame({"started_at": dates})
    base["hour"] = base["started_at"].dt.hour
    base["activity_count"] = (
        25
        + 15 * (base["hour"].between(8, 18).astype(int))
        + 12 * (base["started_at"].dt.dayofweek.isin([5, 6]).astype(int))
        + 18 * ((base["started_at"].dt.month % 2) == 0).astype(int)
        + (base["started_at"].dt.hour % 5)
    )
    base["date"] = base["started_at"].dt.floor("D")
    return base[["date", "hour", "activity_count"]]


def _latest_divvy_zips(max_files: int = 12) -> list[str]:
    response = requests.get(DIVVY_BUCKET_URL, timeout=60)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [
        child.findtext("s3:Key", namespaces=ns)
        for child in root.findall("s3:Contents", ns)
    ]
    keys = [key for key in keys if key and key.endswith(".zip") and "divvy-tripdata" in key.lower()]
    return sorted(keys)[-max_files:]


def _download_divvy_zip(key: str) -> Path:
    archive_path = RAW_DIR / key
    if archive_path.exists():
        return archive_path
    url = f"{DIVVY_BUCKET_URL}{key}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive_path.write_bytes(response.content)
    return archive_path


def _extract_trip_csvs() -> list[Path]:
    csv_paths: list[Path] = []
    for zip_path in sorted(RAW_DIR.glob("*divvy-tripdata*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(RAW_DIR)
        for csv_file in RAW_DIR.glob("*.csv"):
            if csv_file.name.startswith("."):
                continue
            csv_paths.append(csv_file)
    return csv_paths


def _read_real_divvy_data() -> pd.DataFrame:
    zip_keys = _latest_divvy_zips(max_files=6)
    for key in zip_keys:
        _download_divvy_zip(key)
    csv_paths = _extract_trip_csvs()
    if not csv_paths:
        return generate_synthetic_divvy_data()
    frames = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
            if {"started_at", "ended_at"}.issubset(frame.columns):
                frame = frame[["started_at", "ended_at"]].copy()
                frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
                frame["ended_at"] = pd.to_datetime(frame["ended_at"], errors="coerce")
                frame = frame.dropna(subset=["started_at"]).copy()
                frame["date"] = frame["started_at"].dt.floor("D")
                frame["hour"] = frame["started_at"].dt.hour
                frame["activity_count"] = 1
                frames.append(frame[["date", "hour", "activity_count"]])
        except Exception:
            continue
    if not frames:
        return generate_synthetic_divvy_data()
    return pd.concat(frames, ignore_index=True)


def load_divvy_dataset() -> pd.DataFrame:
    try:
        return _read_real_divvy_data()
    except Exception:
        return generate_synthetic_divvy_data()
