"""
UniDomain Data Downloader
=========================

This script downloads the required datasets from Hugging Face based on the specified mode.
It uses 'huggingface_hub' to efficiently fetch only the necessary files and preserve
the directory structure.

Usage:
    python scripts/download_data.py --mode <mode>, or
    unidomain download <mode>

Modes:
    - tutorial:    Videos for pre-training tutorial.
    - meta:        Pre-fused meta-domain and overview task image for planning demo.
    - tasks:       The UniDomain-100 benchmark tasks (images and config).
    - results:     Evaluation logs and metrics for UniDomain and baselines.
    - unified:     The large-scale dataset containing 13k atomic domains and unified domain.
    - all:         Download everything listed above.
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

# Check dependency
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("\n\033[91m[Error] 'huggingface_hub' library is missing.\033[0m")
    print("Please install it via: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "SII-PrimoButterfly/UniDomain-Data"
LOCAL_DIR = Path("data")

# Download patterns for tutorial / meta mode (no compression)
MODE_PATTERNS = {
    "tutorial": ["tutorial/**"],
    "meta":     ["meta_domain/**"],
}

def download_with_snapshot(mode: str):
    """For tutorial / meta download."""

    print(f"\n--- Downloading UniDomain Data (Mode: {mode}) ---")
    print(f"Source Repo : https://huggingface.co/datasets/{REPO_ID}")
    print(f"Destination : {os.path.abspath(LOCAL_DIR)}")
    print("-------------------------------------------------------")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=MODE_PATTERNS[mode],
        resume_download=True,
        tqdm_class=None,
    )


def download_and_extract_tar(remote_filename: str, target_subdir: str):
    """For tasks / results / unified download."""

    print(f"\n--- Downloading archive '{remote_filename}' ---")
    print(f"Source Repo : https://huggingface.co/datasets/{REPO_ID}")
    print(f"Destination : {os.path.abspath(LOCAL_DIR)}")
    print("-------------------------------------------------------")

    # Download to HF cache
    tar_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=remote_filename,
    )
    print(f"Downloaded archive cached at: {tar_path}")


    # Extract to local directory
    target_dir = LOCAL_DIR / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting archive to: {target_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    print(f"\033[92m[Success] Extracted '{remote_filename}' to '{target_dir}/'.\033[0m")


def download_mode(mode: str):
    if not LOCAL_DIR.exists():
        LOCAL_DIR.mkdir(parents=True)

    if mode in ("tutorial", "meta"):
        download_with_snapshot(mode)
        if mode == "meta":
            print(f"Hint: You can find 'meta_domain.pddl' in {LOCAL_DIR}/meta_domain/")
    elif mode == "tasks":
        download_and_extract_tar(
            remote_filename="tasks/tasks.tar.gz",
            target_subdir="tasks",
        )
        print(f"Hint: Task configurations are in {LOCAL_DIR}/tasks/")
    elif mode == "unified":
        download_and_extract_tar(
            remote_filename="unified_domain/unified_domain.tar.gz",
            target_subdir="unified_domain",
        )
    elif mode == "results":
        download_and_extract_tar(
            remote_filename="results/results.tar.gz",
            target_subdir="results",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Download UniDomain datasets.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tutorial", "meta", "tasks", "unified", "results", "all"],
        help="Select the data subset to download."
    )
    args = parser.parse_args()

    if args.mode == "all":
        for m in ["tutorial", "meta", "tasks", "unified", "results"]:
            download_mode(m)
        print(f"\n\033[92m[Success] All data are ready in '{LOCAL_DIR}/'.\033[0m")
    else:
        download_mode(args.mode)


if __name__ == "__main__":
    main()
