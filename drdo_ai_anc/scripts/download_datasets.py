#!/usr/bin/env python3
"""
download_datasets.py — Download verified public audio datasets.

Downloads a manageable first-run subset (~1–3 GB) from Zenodo.
CHiME-3 requires manual registration; clear instructions are printed.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset demand
    python scripts/download_datasets.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# ---------------------------------------------------------------------------
# Dataset registry — verified Zenodo URLs (API content links)
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, Any]] = {
    "chime3": {
        "name": "CHiME-3",
        "license": "Academic — registration required",
        "source_url": "https://www.chimechallenge.org/challenges/chime3",
        "auto_download": False,
        "local_path": RAW_DIR / "chime3",
        "instructions": (
            "\n  CHiME-3 REQUIRES MANUAL DOWNLOAD:\n"
            "  1. Visit: https://www.chimechallenge.org/challenges/chime3\n"
            "  2. Register and accept the license agreement.\n"
            "  3. Download the development/test audio packages.\n"
            "  4. Extract WAV files into: data/raw/chime3/\n"
            "     Expected structure: data/raw/chime3/<environment>/*.wav\n"
        ),
    },
    "demand": {
        "name": "DEMAND",
        "license": "CC BY-SA 4.0",
        "source_url": "https://zenodo.org/record/1227121",
        "auto_download": True,
        "local_path": RAW_DIR / "demand",
        # First-run subset: 4 environments at 16 kHz (~430 MB compressed)
        "files": [
            {
                "name": "NRIVER_16k.zip",
                "url": "https://zenodo.org/api/records/1227121/files/NRIVER_16k.zip/content",
                "md5": "54264db61d3fe073fb81f2e40e0d19b5",
                "size": 98689565,
            },
            {
                "name": "TBUS_16k.zip",
                "url": "https://zenodo.org/api/records/1227121/files/TBUS_16k.zip/content",
                "md5": "706b11b0d8504f9f3b3f3211e91b3863",
                "size": 128916709,
            },
            {
                "name": "OOFFICE_16k.zip",
                "url": "https://zenodo.org/api/records/1227121/files/OOFFICE_16k.zip/content",
                "md5": "7b61cc2d182d5a654cb9c3101ddd4041",
                "size": 88995191,
            },
            {
                "name": "STRAFFIC_16k.zip",
                "url": "https://zenodo.org/api/records/1227121/files/STRAFFIC_16k.zip/content",
                "md5": "2efa87262f272bbf9ba578088e81939c",
                "size": 118572691,
            },
        ],
    },
    "sonyc_ust": {
        "name": "SONYC-UST",
        "license": "CC BY 4.0",
        "source_url": "https://zenodo.org/record/3966543",
        "auto_download": True,
        "local_path": RAW_DIR / "sonyc_ust",
        "files": [
            {
                "name": "annotations.csv",
                "url": "https://zenodo.org/api/records/3966543/files/annotations.csv/content",
                "md5": "70b507b15bb4cfcce4870925302f276b",
                "size": 14488305,
            },
            {
                "name": "audio-18.tar.gz",
                "url": "https://zenodo.org/api/records/3966543/files/audio-18.tar.gz/content",
                "md5": "c6b2ef4d0d5b7269d465c469cdbbdc4b",
                "size": 365905437,
            },
        ],
    },
}


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected_md5: Optional[str] = None) -> None:
    """Download with progress; skip if valid file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_md5:
            actual = _md5_file(dest)
            if actual == expected_md5:
                print(f"  SKIP (already valid): {dest.name}")
                return
            print(f"  RE-DOWNLOAD (checksum mismatch): {dest.name}")
        else:
            print(f"  SKIP (exists): {dest.name}")
            return

    print(f"  DOWNLOADING: {dest.name}")
    with urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block = 1 << 16
        with open(dest, "wb") as out:
            while True:
                data = response.read(block)
                if not data:
                    break
                out.write(data)
                downloaded += len(data)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(
                        f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()

    if expected_md5:
        actual = _md5_file(dest)
        if actual != expected_md5:
            raise RuntimeError(
                f"Checksum failed for {dest.name}: expected {expected_md5}, got {actual}"
            )
        print(f"  VERIFIED: {dest.name}")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  SKIP extract (already done): {zip_path.name}")
        return
    print(f"  EXTRACTING: {zip_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.touch()


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / f".extracted_{tar_path.stem}"
    if marker.exists():
        print(f"  SKIP extract (already done): {tar_path.name}")
        return
    print(f"  EXTRACTING: {tar_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    marker.touch()


def _count_audio_files(directory: Path) -> int:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)


def check_dataset(key: str) -> bool:
    info = DATASETS[key]
    local = info["local_path"]
    return _count_audio_files(local) > 0


def download_dataset(key: str) -> Dict[str, Any]:
    """Download and extract one dataset. Returns summary dict."""
    info = DATASETS[key]
    summary: Dict[str, Any] = {
        "dataset": key,
        "name": info["name"],
        "license": info["license"],
        "source_url": info["source_url"],
        "status": "skipped",
        "files_downloaded": 0,
        "audio_files": 0,
    }

    print(f"\n{'='*60}")
    print(f"  {info['name']}")
    print(f"  License: {info['license']}")
    print(f"  Source:  {info['source_url']}")
    print(f"{'='*60}")

    if not info.get("auto_download", False):
        print(info["instructions"])
        summary["status"] = "manual_required"
        return summary

    local = info["local_path"]
    archives_dir = local / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for fmeta in info.get("files", []):
        archive_path = archives_dir / fmeta["name"]
        try:
            _download_file(fmeta["url"], archive_path, fmeta.get("md5"))
            downloaded += 1

            if fmeta["name"].endswith(".zip"):
                _extract_zip(archive_path, local)
            elif fmeta["name"].endswith((".tar.gz", ".tgz")):
                _extract_tar(archive_path, local)
            # annotations.csv stays in archives_dir, copy to local root
            if fmeta["name"] == "annotations.csv":
                import shutil

                dest = local / "annotations.csv"
                if not dest.exists():
                    shutil.copy2(archive_path, dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", fmeta["name"], exc)
            summary["status"] = "error"
            summary["error"] = str(exc)
            return summary

    audio_count = _count_audio_files(local)
    summary["files_downloaded"] = downloaded
    summary["audio_files"] = audio_count
    summary["status"] = "downloaded" if audio_count > 0 else "no_audio_found"
    print(f"\n  Audio files found: {audio_count}")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download real audio datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        print("\n  Dataset presence check:")
        for k in DATASETS:
            ok = check_dataset(k)
            n = _count_audio_files(DATASETS[k]["local_path"])
            status = f"FOUND ({n} files)" if ok else "MISSING"
            print(f"  {k:<15} {status}")
        return 0

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())
    reports: List[Dict[str, Any]] = []

    for key in targets:
        reports.append(download_dataset(key))

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METADATA_DIR / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)

    print(f"\n  Download report saved: {report_path}")
    total_audio = sum(r.get("audio_files", 0) for r in reports)
    print(f"  Total audio files across datasets: {total_audio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
