"""
hf_loader.py
------------
Downloads model and data artefacts from a Hugging Face Hub dataset repo
at application startup if they are not already present on disk.

Environment variables
---------------------
HF_REPO_ID   : The HF dataset repo that holds the large files.
               Default: "biplobgon/product-recommendation-data"
HF_TOKEN     : HF access token (only needed if the repo is private).
               On Hugging Face Spaces, set this in the Space's Secrets panel.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

HF_REPO_ID: str = os.getenv("HF_REPO_ID", "biplobgon/product-recommendation-data")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Files that must be present for the app to function correctly.
# Mapping: repo path (relative inside the HF dataset) -> local destination.
_ARTEFACTS: dict[str, Path] = {
    # --- model weights -------------------------------------------------
    "models/als_model.pkl":           ROOT / "outputs" / "models" / "als_model.pkl",
    "models/session_based_model.pkl": ROOT / "outputs" / "models" / "session_based_model.pkl",
    "models/content_based_model.pkl": ROOT / "outputs" / "models" / "content_based_model.pkl",
    "models/hybrid_model.pkl":        ROOT / "outputs" / "models" / "hybrid_model.pkl",
    # --- processed data -------------------------------------------------
    "processed/user_features.parquet":   ROOT / "data" / "processed" / "user_features.parquet",
    "processed/item_features.parquet":   ROOT / "data" / "processed" / "item_features.parquet",
    "processed/interactions.csv":        ROOT / "data" / "processed" / "interactions.csv",
    # --- eval report ----------------------------------------------------
    "reports/evaluation_report.csv":     ROOT / "outputs" / "reports" / "evaluation_report.csv",
    # --- raw reference data ---------------------------------------------
    "raw/category_tree.csv":             ROOT / "data" / "raw" / "category_tree.csv",
}


def ensure_all(show_progress: bool = True) -> None:
    """Download any missing artefacts from the HF dataset repo.

    Safe to call multiple times — already-present files are skipped.
    On a plain local machine (where files already exist) this is a no-op.
    """
    missing = {repo_path: local for repo_path, local in _ARTEFACTS.items() if not local.exists()}
    if not missing:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _warn(
            "huggingface_hub is not installed. "
            "Run `pip install huggingface_hub` or add it to requirements.txt."
        )
        return

    if show_progress:
        try:
            import streamlit as st
            progress_fn = st.toast
        except Exception:
            progress_fn = print
    else:
        progress_fn = lambda _: None  # noqa: E731

    for repo_path, local_path in missing.items():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if show_progress:
                progress_fn(f"Downloading {local_path.name} …")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=str(ROOT),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to local_dir/<filename>; move to expected path
            dl_path = Path(downloaded)
            if dl_path.resolve() != local_path.resolve():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                dl_path.rename(local_path)
        except Exception as exc:
            _warn(f"Could not download {repo_path}: {exc}")


def _warn(msg: str) -> None:
    try:
        import streamlit as st
        st.warning(msg)
    except Exception:
        print(f"[hf_loader] WARNING: {msg}")
