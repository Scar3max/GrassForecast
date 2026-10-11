from __future__ import annotations

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
    csv_paths: set[Path] = set(RAW_DIR.glob("*divvy-tripdata*.csv"))
    for zip_path in sorted(RAW_DIR.glob("*divvy-tripdata*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                path = Path(member)
                if path.suffix.lower() != ".csv" or path.name.startswith("."):
                    continue
                extracted = RAW_DIR / path.name
                if not extracted.exists():
                    with archive.open(member) as source, extracted.open("wb") as target:
                        target.write(source.read())
                csv_paths.add(extracted)
    return sorted(csv_paths)


def _aggregate_trip_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    if not csv_paths:
        raise RuntimeError("No Divvy trip CSVs were downloaded or found in data/raw.")
    frames = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path, usecols=["started_at"])
            frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
            frame = frame.dropna(subset=["started_at"])
            frame["date"] = frame["started_at"].dt.floor("D")
            frame["hour"] = frame["started_at"].dt.hour
            frame["activity_count"] = 1
            hourly_counts = frame.groupby(["date", "hour"], as_index=False)["activity_count"].sum()
            frames.append(hourly_counts)
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            print(f"Skipping unreadable Divvy file {csv_path.name}: {exc}")
            continue
    if not frames:
        raise RuntimeError("Divvy files were found, but none contained readable started_at timestamps.")
    return pd.concat(frames, ignore_index=True).groupby(
        ["date", "hour"], as_index=False
    )["activity_count"].sum()


def _read_real_divvy_data() -> pd.DataFrame:
    zip_keys = _latest_divvy_zips(max_files=18)
    for key in zip_keys:
        _download_divvy_zip(key)
    return _aggregate_trip_csvs(_extract_trip_csvs())


def load_divvy_dataset() -> pd.DataFrame:
    try:
        return _read_real_divvy_data()
    except requests.RequestException as exc:
        cached_csvs = _extract_trip_csvs()
        if cached_csvs:
            print(f"Divvy download failed; using cached trip files: {exc}")
            return _aggregate_trip_csvs(cached_csvs)
        raise RuntimeError(f"Could not download Divvy activity data: {exc}") from exc
