"""
data_prep.py
------------
Load and inspect the raw Retailrocket events dataset.

Data source priority:
  1. Local file at data/raw/events.csv  (fast path – no network required)
  2. Google Cloud Storage bucket        (auto-downloaded when local file is absent)

Set GCS_BUCKET_NAME (and optionally GCS_CREDENTIALS / GCS_PROJECT_ID) in a
.env file or as environment variables to configure the GCS connection.
See .env.example for details.
"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Resolve project root so the script works when called from any directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
EVENTS_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "events.csv")


def load_events() -> pd.DataFrame:
    """Return the events DataFrame, fetching from GCS if the file is missing."""
    if not os.path.exists(EVENTS_PATH):
        print(f"Local events file not found at {EVENTS_PATH}.")
        print("Attempting to download from Google Cloud Storage …")
        try:
            from gcs_loader import load_events as _gcs_load  # noqa: PLC0415
        except ImportError as exc:
            print(f"ERROR: Could not import gcs_loader: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            return _gcs_load(local_path=EVENTS_PATH)
        except FileNotFoundError as exc:
            print(f"ERROR: Dataset file not found in GCS: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            # Covers google.auth.exceptions.DefaultCredentialsError and other
            # GCS / network failures with a helpful remediation hint.
            print(f"ERROR: Could not download events data from GCS: {exc}", file=sys.stderr)
            print(
                "Please ensure:\n"
                "  1. GCS_BUCKET_NAME is set correctly (see .env.example).\n"
                "  2. You are authenticated (gcloud auth application-default login\n"
                "     or GCS_CREDENTIALS points to a valid service-account key).",
                file=sys.stderr,
            )
            sys.exit(1)

    return pd.read_csv(EVENTS_PATH)


if __name__ == "__main__":
    events = load_events()

    print(events.head())
    print(events.info())
    print(events["event"].value_counts())
