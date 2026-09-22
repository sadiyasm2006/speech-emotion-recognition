"""
Feature extraction module for Speech Emotion Recognition.

Extracts fixed-length MFCC statistical features (mean and std dev)
from audio files using librosa.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa


def extract_features(
    file_path: Union[str, Path],
    n_mfcc: int = 40,
    duration: float = 3.5,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extracts an 80-dimensional MFCC feature vector from an audio file.

    Parameters:
        file_path: Path to the .wav audio file.
        n_mfcc: Number of MFCC coefficients to compute (default: 40).
        duration: Target duration in seconds to pad/truncate to (default: 3.5).
        sr: Sampling rate to load audio (default: 16000 Hz).

    Returns:
        np.ndarray: Concatenated mean and standard deviation of MFCCs, shape (80,).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If audio loading or feature extraction fails.
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    try:
        # 1. Load audio at specified sampling rate
        # librosa.load returns (y, sr)
        y, sample_rate = librosa.load(str(path_obj), sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio from {file_path}: {e}") from e

    if y is None or len(y) == 0:
        raise ValueError(f"Audio file is empty or corrupted: {file_path}")

    # 2. Limit/pad audio signal to target duration (3.5s @ 16kHz = 56000 samples)
    target_length = int(duration * sr)
    if len(y) < target_length:
        # Zero-pad at the end
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        # Truncate to target length
        y = y[:target_length]

    try:
        # 3. Extract MFCCs across time frames
        # mfccs shape: (n_mfcc, t)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 4. Calculate mean and standard deviation across time axis (axis=1)
        mfcc_mean = np.mean(mfccs, axis=1)  # shape: (n_mfcc,)
        mfcc_std = np.std(mfccs, axis=1)    # shape: (n_mfcc,)

        # 5. Concatenate into a single feature vector of shape (2 * n_mfcc,) = (80,)
        feature_vector = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)

        return feature_vector

    except Exception as e:
        raise ValueError(f"Feature computation failed for {file_path}: {e}") from e


if __name__ == "__main__":
    import sys

    # Test run on a sample file
    base_dir = Path(__file__).resolve().parent.parent
    sample_files = list(base_dir.rglob("data/ravdess/**/*.wav"))

    if not sample_files:
        sample_files = list(base_dir.rglob("data/cremad/*.wav"))

    if sample_files:
        test_file = sample_files[0]
        print(f"Testing feature extraction on: {test_file}")
        try:
            feats = extract_features(test_file)
            print(f"Extraction successful!")
            print(f"Feature Vector Shape: {feats.shape}")
            print(f"Feature Vector Dtype: {feats.dtype}")
            print(f"First 5 values: {feats[:5]}")
        except Exception as err:
            print(f"Extraction failed with error: {err}")
            sys.exit(1)
    else:
        print("No audio files found for testing.")
