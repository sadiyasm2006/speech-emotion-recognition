"""
Dataset preparation script for Domain-Adaptive Speech Emotion Recognition.

Parses RAVDESS (source domain) and CREMA-D (target domain) audio datasets,
extracts emotion labels, validates .wav files, and selects a reproducible
subset of ~200 files per dataset for development and prototyping.
"""

import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"
CREMAD_DIR = DATA_DIR / "cremad"

# Emotion mappings
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

CREMAD_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_ravdess_filename(file_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse a RAVDESS audio filename.
    Format: 03-01-06-01-02-01-12.wav
    Parts: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    """
    stem = file_path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None

    emotion_code = parts[2]
    emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
    if not emotion:
        return None

    return {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "dataset": "RAVDESS",
        "domain": "source",
        "emotion_code": emotion_code,
        "emotion": emotion,
        "actor_id": parts[6],
    }


def parse_cremad_filename(file_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse a CREMA-D audio filename.
    Format: 1001_DFA_ANG_XX.wav
    Parts: ActorID_Sentence_Emotion_Intensity.wav
    """
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    emotion_code = parts[2].upper()
    emotion = CREMAD_EMOTION_MAP.get(emotion_code)
    if not emotion:
        return None

    return {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "dataset": "CREMA-D",
        "domain": "target",
        "emotion_code": emotion_code,
        "emotion": emotion,
        "actor_id": parts[0],
    }


def scan_dataset(
    dataset_dir: Path,
    dataset_type: str,
) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Recursively scans a dataset folder for .wav files and parses their labels.

    Returns:
        valid_records: List of parsed metadata dictionaries
        total_wav_found: Total count of .wav files found
        skipped_count: Count of files that failed validation / parsing
    """
    if not dataset_dir.exists():
        return [], 0, 0

    wav_files = list(dataset_dir.rglob("*.wav")) + list(dataset_dir.rglob("*.WAV"))
    # Remove duplicates if case-insensitive filesystem returns both
    wav_files = sorted(list({f.resolve(): f for f in wav_files}.values()))

    valid_records = []
    skipped_count = 0

    for file_path in wav_files:
        if dataset_type.lower() == "ravdess":
            record = parse_ravdess_filename(file_path)
        elif dataset_type.lower() == "cremad":
            record = parse_cremad_filename(file_path)
        else:
            record = None

        if record:
            valid_records.append(record)
        else:
            skipped_count += 1

    return valid_records, len(wav_files), skipped_count


def select_reproducible_subset(
    records: List[Dict[str, str]],
    target_count: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Selects a balanced, reproducible subset of audio files stratified by emotion.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if len(df) <= target_count:
        return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Stratified sampling across emotions to ensure balanced representation
    emotions = df["emotion"].unique()
    samples_per_emotion = target_count // len(emotions)
    remainder = target_count % len(emotions)

    sampled_dfs = []
    for i, (emotion, group) in enumerate(df.groupby("emotion")):
        n_sample = samples_per_emotion + (1 if i < remainder else 0)
        n_sample = min(n_sample, len(group))
        sampled_dfs.append(group.sample(n=n_sample, random_state=random_seed))

    subset_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return subset_df


def prepare_data_pipeline(
    target_per_dataset: int = 200,
    random_seed: int = 42,
    save_manifest: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes dataset scanning, statistics reporting, and subset selection.
    """
    print("=" * 60)
    print("SPEECH EMOTION CORAL - DATASET PREPARATION PIPELINE")
    print("=" * 60)

    # 1. Scan RAVDESS
    ravdess_records, ravdess_wavs, ravdess_skipped = scan_dataset(RAVDESS_DIR, "ravdess")
    print(f"\n[1] RAVDESS (Source Domain):")
    print(f"    - Directory: {RAVDESS_DIR}")
    print(f"    - Total WAV files found: {ravdess_wavs}")
    print(f"    - Valid parsed files:    {len(ravdess_records)}")
    print(f"    - Skipped/Invalid files: {ravdess_skipped}")

    ravdess_subset = select_reproducible_subset(
        ravdess_records, target_count=target_per_dataset, random_seed=random_seed
    )

    if not ravdess_subset.empty:
        print(f"    - Selected subset size:  {len(ravdess_subset)}")
        print("    - Emotion distribution:")
        for emo, count in ravdess_subset["emotion"].value_counts().items():
            print(f"        * {emo:<10}: {count}")
    else:
        print("    - [!] No valid RAVDESS audio files found. Please populate data/ravdess/")

    # 2. Scan CREMA-D
    cremad_records, cremad_wavs, cremad_skipped = scan_dataset(CREMAD_DIR, "cremad")
    print(f"\n[2] CREMA-D (Target Domain):")
    print(f"    - Directory: {CREMAD_DIR}")
    print(f"    - Total WAV files found: {cremad_wavs}")
    print(f"    - Valid parsed files:    {len(cremad_records)}")
    print(f"    - Skipped/Invalid files: {cremad_skipped}")

    cremad_subset = select_reproducible_subset(
        cremad_records, target_count=target_per_dataset, random_seed=random_seed
    )

    if not cremad_subset.empty:
        print(f"    - Selected subset size:  {len(cremad_subset)}")
        print("    - Emotion distribution:")
        for emo, count in cremad_subset["emotion"].value_counts().items():
            print(f"        * {emo:<10}: {count}")
    else:
        print("    - [!] No valid CREMA-D audio files found. Please populate data/cremad/")

    # 3. Save manifests if data is present
    if save_manifest and (not ravdess_subset.empty or not cremad_subset.empty):
        if not ravdess_subset.empty:
            ravdess_csv = DATA_DIR / "ravdess_subset.csv"
            ravdess_subset.to_csv(ravdess_csv, index=False)
            print(f"\nSaved RAVDESS subset manifest: {ravdess_csv}")

        if not cremad_subset.empty:
            cremad_csv = DATA_DIR / "cremad_subset.csv"
            cremad_subset.to_csv(cremad_csv, index=False)
            print(f"Saved CREMA-D subset manifest: {cremad_csv}")

    print("\n" + "=" * 60)
    return ravdess_subset, cremad_subset


if __name__ == "__main__":
    prepare_data_pipeline(target_per_dataset=200, random_seed=42, save_manifest=True)
