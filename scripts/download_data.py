"""Download the raw UCI Heart Disease (Cleveland) file and verify its checksum.

Usage:
    python scripts/download_data.py [--force]
"""

import argparse
import hashlib
import sys

import requests

from heart.config import RAW_DATA_PATH, RAW_DATA_SHA256, RAW_DATA_URL


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if RAW_DATA_PATH.exists() and not args.force:
        if sha256(RAW_DATA_PATH.read_bytes()) == RAW_DATA_SHA256:
            print(f"Already downloaded and verified: {RAW_DATA_PATH}")
            return 0
        print("Existing file has the wrong checksum, re-downloading")

    print(f"Downloading {RAW_DATA_URL}")
    response = requests.get(RAW_DATA_URL, timeout=30)
    response.raise_for_status()

    digest = sha256(response.content)
    if digest != RAW_DATA_SHA256:
        print(f"Checksum mismatch: expected {RAW_DATA_SHA256}, got {digest}", file=sys.stderr)
        return 1

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {RAW_DATA_PATH} (sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
