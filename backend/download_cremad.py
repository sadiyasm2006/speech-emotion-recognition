"""
CREMA-D Subset Downloader for Domain-Adaptive Speech Emotion Recognition.

Downloads a reproducible, balanced subset of approximately 200 WAV audio files
from the official CREMA-D repository (CheyneyComputerScience/CREMA-D) via Git LFS media endpoints.
Validates WAV file integrity to ensure real audio is fetched rather than Git LFS pointer files.
"""

import os
from pathlib import Path
import random
import json
import urllib.request
import urllib.error
from typing import Dict, List, Tuple
from collections import Counter

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "cremad"

# Official CREMA-D Repository metadata
REPO_API_TREE_URL = (
    "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
)
RAW_MEDIA_BASE_URL = (
    "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/"
)

# Supported emotions
VALID_EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_NAMES = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def is_valid_wav_header(file_path: Path) -> Tuple[bool, str]:
    """
    Validates that a file is an authentic WAV binary file and not a Git LFS text pointer.
    """
    if not file_path.exists():
        return False, "File does not exist"

    size = file_path.stat().st_size
    if size < 1024:
        # Check if it's an LFS pointer
        with open(file_path, "rb") as f:
            header = f.read(100)
            if b"git-lfs" in header or b"oid sha256" in header:
                return False, "File is a Git LFS pointer text file (not real audio)"
        return False, f"File size too small ({size} bytes)"

    with open(file_path, "rb") as f:
        header = f.read(12)
        if len(header) < 12:
            return False, "Incomplete file header"
        # WAV files start with RIFF....WAVE
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return False, f"Invalid WAV header magic bytes: {header[:4]}...{header[8:12]}"

    return True, "Valid WAV audio"


def fetch_cremad_file_list() -> List[str]:
    """
    Fetches the list of all available WAV files from the official CREMA-D repo.
    """
    print("[1/4] Fetching remote file catalog from official CREMA-D repository...")
    req = urllib.request.Request(
        REPO_API_TREE_URL,
        headers={"User-Agent": "Mozilla/5.0 (Python CREMA-D Subset Downloader)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        tree_data = json.loads(resp.read().decode("utf-8"))

    wav_files = []
    for item in tree_data.get("tree", []):
        path_str = item.get("path", "")
        if path_str.startswith("AudioWAV/") and path_str.lower().endswith(".wav"):
            file_name = path_str.split("/")[-1]
            wav_files.append(file_name)

    print(f"      Found {len(wav_files)} total audio files in repository catalog.")
    return sorted(wav_files)


def select_balanced_subset(
    all_files: List[str], target_total: int = 200, seed: int = 42
) -> List[str]:
    """
    Groups available files by emotion and selects a balanced reproducible subset.
    """
    grouped_by_emotion: Dict[str, List[str]] = {emo: [] for emo in VALID_EMOTIONS}

    for fname in all_files:
        parts = fname.split("_")
        if len(parts) >= 3:
            emo = parts[2].upper()
            if emo in grouped_by_emotion:
                grouped_by_emotion[emo].append(fname)

    rng = random.Random(seed)
    selected_files: List[str] = []

    per_emotion_count = target_total // len(VALID_EMOTIONS)
    remainder = target_total % len(VALID_EMOTIONS)

    for i, emo in enumerate(VALID_EMOTIONS):
        files_for_emo = sorted(grouped_by_emotion[emo])
        quota = per_emotion_count + (1 if i < remainder else 0)
        quota = min(quota, len(files_for_emo))
        sampled = rng.sample(files_for_emo, quota)
        selected_files.extend(sampled)

    # Deterministic shuffle of the full subset
    rng.shuffle(selected_files)
    return selected_files


def download_file(filename: str, output_dir: Path) -> bool:
    """
    Downloads a single WAV file from official Git LFS media storage.
    """
    target_path = output_dir / filename
    if target_path.exists():
        valid, _ = is_valid_wav_header(target_path)
        if valid:
            return True  # Already successfully downloaded

    url = RAW_MEDIA_BASE_URL + filename
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Python CREMA-D Subset Downloader)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(target_path, "wb") as f:
                f.write(content)

        # Immediate verification
        is_valid, reason = is_valid_wav_header(target_path)
        if not is_valid:
            print(f"      [!] Verification failed for {filename}: {reason}")
            if target_path.exists():
                target_path.unlink()
            return False
        return True

    except urllib.error.URLError as e:
        print(f"      [!] Failed to download {filename}: {e}")
        if target_path.exists():
            target_path.unlink()
        return False


def run_downloader(target_total: int = 200, seed: int = 42):
    """
    Main pipeline for downloading the balanced CREMA-D subset.
    """
    print("=" * 65)
    print("CREMA-D BALANCED SUBSET DOWNLOADER")
    print(f"Target Subset Size: {target_total} files | Random Seed: {seed}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch file catalog
    all_files = fetch_cremad_file_list()
    if not all_files:
        print("[!] No audio files retrieved from repository catalog. Aborting.")
        return

    # 2. Select balanced subset
    print(f"\n[2/4] Selecting balanced subset of ~{target_total} files (seed={seed})...")
    selected_files = select_balanced_subset(all_files, target_total=target_total, seed=seed)
    print(f"      Selected {len(selected_files)} candidate files across {len(VALID_EMOTIONS)} emotions.")

    # 3. Download files
    print(f"\n[3/4] Downloading audio files into: {OUTPUT_DIR} ...")
    success_count = 0
    downloaded_emotions = Counter()

    for idx, fname in enumerate(selected_files, start=1):
        emo_code = fname.split("_")[2].upper()
        print(f"      [{idx:03d}/{len(selected_files):03d}] Downloading {fname} ({EMOTION_NAMES.get(emo_code, emo_code)})...", end=" ")
        ok = download_file(fname, OUTPUT_DIR)
        if ok:
            success_count += 1
            downloaded_emotions[emo_code] += 1
            print("DONE (Verified WAV)")
        else:
            print("FAILED")

    # 4. Summary report
    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY & EMOTION DISTRIBUTION")
    print("=" * 65)
    print(f"Destination directory:  {OUTPUT_DIR}")
    print(f"Successfully validated: {success_count} / {len(selected_files)} WAV files")
    print("\nEmotion Breakdown:")
    for emo_code in VALID_EMOTIONS:
        name = EMOTION_NAMES.get(emo_code, emo_code)
        count = downloaded_emotions[emo_code]
        print(f"  * [{emo_code}] {name.capitalize():<10}: {count:3d} files")
    print("=" * 65)


if __name__ == "__main__":
    run_downloader(target_total=200, seed=42)
