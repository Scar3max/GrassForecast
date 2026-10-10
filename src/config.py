from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "output"
NUDGE_DIR = OUTPUT_DIR / "nudges"
CONFIG_DIR = BASE_DIR / "config"

for directory in (RAW_DIR, CACHE_DIR, RESULTS_DIR, NUDGE_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def config_value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)
