"""
gcs_loader.py
-------------
Utility module to download dataset files from a Google Cloud Storage bucket.

Configuration (via environment variables or .env file):
    GCS_BUCKET_NAME   : Name of the GCS bucket (default: product-recommender-systems)
    GCS_PROJECT_ID    : GCP project ID (optional – inferred from ADC when omitted)
    GCS_CREDENTIALS   : Path to a service-account JSON key file (optional –
                        falls back to Application Default Credentials)
    DATA_RAW_DIR      : Local directory where raw files are saved
                        (default: data/raw)
"""

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.exceptions import NotFound

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "product-recommender-systems")
PROJECT_ID: str | None = os.getenv("GCS_PROJECT_ID")
CREDENTIALS_PATH: str | None = os.getenv("GCS_CREDENTIALS")
DATA_RAW_DIR: str = os.getenv("DATA_RAW_DIR", "data/raw")

# Files that make up the Retailrocket dataset stored in the bucket
DEFAULT_FILES: list[str] = [
    "events.csv",
    "category_tree.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _get_client() -> storage.Client:
    """Return an authenticated GCS client.

    Authentication priority:
    1. Service-account key file pointed to by GCS_CREDENTIALS env var.
    2. Application Default Credentials (``gcloud auth application-default login``
       or a workload-identity / service-account attached to the compute instance).
    """
    kwargs: dict = {}
    if PROJECT_ID:
        kwargs["project"] = PROJECT_ID
    if CREDENTIALS_PATH:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH
        )
        kwargs["credentials"] = credentials

    return storage.Client(**kwargs)


def download_from_gcs(
    source_blob_name: str,
    destination_file_name: str,
    bucket_name: str = BUCKET_NAME,
    overwrite: bool = False,
) -> str:
    """Download a single file from GCS to the local filesystem.

    Parameters
    ----------
    source_blob_name:
        Path of the object inside the bucket (e.g. ``"events.csv"``).
    destination_file_name:
        Local path where the file should be saved.
    bucket_name:
        GCS bucket name. Defaults to ``GCS_BUCKET_NAME`` env var.
    overwrite:
        When *False* (default) the download is skipped if the destination
        file already exists.

    Returns
    -------
    str
        Absolute path of the downloaded (or already existing) file.
    """
    destination_file_name = os.path.abspath(destination_file_name)

    if not overwrite and os.path.exists(destination_file_name):
        logger.info("File already exists, skipping download: %s", destination_file_name)
        return destination_file_name

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    try:
        blob.download_to_filename(destination_file_name)
        logger.info("Downloaded gs://%s/%s → %s", bucket_name, source_blob_name, destination_file_name)
    except NotFound:
        raise FileNotFoundError(
            f"Object not found in GCS: gs://{bucket_name}/{source_blob_name}"
        )

    return destination_file_name


def download_dataset(
    files: list[str] | None = None,
    local_dir: str = DATA_RAW_DIR,
    bucket_name: str = BUCKET_NAME,
    overwrite: bool = False,
) -> dict[str, str]:
    """Download one or more dataset files from GCS.

    Parameters
    ----------
    files:
        List of blob names to download. Defaults to :data:`DEFAULT_FILES`.
    local_dir:
        Local directory where files are saved.
    bucket_name:
        GCS bucket name.
    overwrite:
        Re-download even if the local file already exists.

    Returns
    -------
    dict[str, str]
        Mapping of ``blob_name → local_path`` for every successfully
        downloaded file.
    """
    if files is None:
        files = DEFAULT_FILES

    results: dict[str, str] = {}
    for blob_name in files:
        local_path = os.path.join(local_dir, os.path.basename(blob_name))
        try:
            path = download_from_gcs(
                source_blob_name=blob_name,
                destination_file_name=local_path,
                bucket_name=bucket_name,
                overwrite=overwrite,
            )
            results[blob_name] = path
        except FileNotFoundError as exc:
            logger.warning("Skipping missing file: %s", exc)

    return results


def load_events(
    local_path: str | None = None,
    bucket_name: str = BUCKET_NAME,
    blob_name: str = "events.csv",
) -> pd.DataFrame:
    """Load the events CSV, downloading from GCS first if needed.

    Parameters
    ----------
    local_path:
        Where to store / look for the local copy.  Defaults to
        ``<DATA_RAW_DIR>/events.csv``.
    bucket_name:
        GCS bucket name.
    blob_name:
        Object name inside the bucket.

    Returns
    -------
    pd.DataFrame
        The events data frame with columns:
        ``timestamp``, ``visitorid``, ``event``, ``itemid``, ``transactionid``.
    """
    if local_path is None:
        local_path = os.path.join(DATA_RAW_DIR, "events.csv")

    if not os.path.exists(local_path):
        logger.info("events.csv not found locally – fetching from GCS …")
        download_from_gcs(blob_name, local_path, bucket_name=bucket_name)

    df = pd.read_csv(local_path)
    logger.info("Loaded events data: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_category_tree(
    local_path: str | None = None,
    bucket_name: str = BUCKET_NAME,
    blob_name: str = "category_tree.csv",
) -> pd.DataFrame:
    """Load the category tree CSV, downloading from GCS first if needed."""
    if local_path is None:
        local_path = os.path.join(DATA_RAW_DIR, "category_tree.csv")

    if not os.path.exists(local_path):
        logger.info("category_tree.csv not found locally – fetching from GCS …")
        download_from_gcs(blob_name, local_path, bucket_name=bucket_name)

    df = pd.read_csv(local_path)
    logger.info("Loaded category tree: %d rows, %d columns", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download dataset files from GCS.")
    parser.add_argument(
        "--bucket",
        default=BUCKET_NAME,
        help=f"GCS bucket name (default: {BUCKET_NAME})",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Blob names to download (default: all dataset files)",
    )
    parser.add_argument(
        "--dest",
        default=DATA_RAW_DIR,
        help=f"Local destination directory (default: {DATA_RAW_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if local file already exists",
    )
    args = parser.parse_args()

    downloaded = download_dataset(
        files=args.files,
        local_dir=args.dest,
        bucket_name=args.bucket,
        overwrite=args.overwrite,
    )

    if downloaded:
        print("\nDownloaded files:")
        for blob, path in downloaded.items():
            print(f"  {blob} → {path}")
    else:
        print("No files were downloaded.")