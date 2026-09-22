"""
Speech Emotion Recognition Baseline & CORAL Domain Adaptation Pipeline.

Trains a Random Forest classifier on RAVDESS (source domain) and evaluates
cross-dataset generalization on CREMA-D (target domain) before and after
CORAL domain adaptation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from backend.features import extract_features
from backend.coral import CORAL

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"
CREMAD_DIR = DATA_DIR / "cremad"
MODEL_PATH = BASE_DIR / "backend" / "model.joblib"
CORAL_PATH = BASE_DIR / "backend" / "coral_reference.npz"

# 6 Common shared emotions across RAVDESS and CREMA-D
SHARED_EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]

# Emotion mappings
RAVDESS_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}

CREMAD_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def load_dataset_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans data/ravdess and data/cremad recursively for audio files
    with shared emotion labels.
    """
    # 1. Scan RAVDESS
    ravdess_files = sorted(list(RAVDESS_DIR.rglob("*.wav")) + list(RAVDESS_DIR.rglob("*.WAV")))
    ravdess_records = []
    for p in ravdess_files:
        stem = p.stem
        parts = stem.split("-")
        if len(parts) == 7:
            emo_code = parts[2]
            if emo_code in RAVDESS_MAP:
                ravdess_records.append({
                    "file_path": str(p),
                    "file_name": p.name,
                    "dataset": "RAVDESS",
                    "emotion": RAVDESS_MAP[emo_code],
                })

    ravdess_df = pd.DataFrame(ravdess_records)

    # 2. Scan CREMA-D
    cremad_files = sorted(list(CREMAD_DIR.rglob("*.wav")) + list(CREMAD_DIR.rglob("*.WAV")))
    cremad_records = []
    for p in cremad_files:
        stem = p.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            emo_code = parts[2].upper()
            if emo_code in CREMAD_MAP:
                cremad_records.append({
                    "file_path": str(p),
                    "file_name": p.name,
                    "dataset": "CREMA-D",
                    "emotion": CREMAD_MAP[emo_code],
                })

    cremad_df = pd.DataFrame(cremad_records)
    return ravdess_df, cremad_df


def extract_features_for_df(df: pd.DataFrame, desc: str = "Dataset") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 80-dim MFCC features for each audio file in a metadata DataFrame.
    """
    features_list = []
    labels_list = []
    total = len(df)

    print(f"Extracting features for {desc} ({total} audio files)...")
    for idx, row in df.iterrows():
        try:
            feat = extract_features(row["file_path"], n_mfcc=40, duration=3.5, sr=16000)
            features_list.append(feat)
            labels_list.append(row["emotion"])
        except Exception as e:
            print(f"  [!] Skipped {row['file_name']}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    print(f"  Extracted {len(X)} feature vectors of shape {X.shape[1:]} for {desc}.")
    return X, y


def train_and_evaluate_pipeline(
    random_state: int = 42,
    coral_reg: float = 1e-4,
) -> Dict[str, Union[float, int, str]]:
    """
    Full workflow:
    1. Extract features for RAVDESS and CREMA-D.
    2. Split RAVDESS into train (80%) and test (20%).
    3. Split CREMA-D into reference pool (50%) and evaluation set (50%).
    4. Train RandomForest on RAVDESS train set.
    5. Evaluate within-domain RAVDESS test accuracy.
    6. Fit CORAL between RAVDESS train features and CREMA-D reference features.
    7. Evaluate CREMA-D evaluation set BEFORE and AFTER CORAL.
    8. Save model and CORAL reference parameters.
    """
    print("=" * 70)
    print("SPEECH EMOTION RECOGNITION & CORAL DOMAIN ADAPTATION TRAINING")
    print("=" * 70)

    # 1. Load file metadata
    ravdess_df, cremad_df = load_dataset_metadata()
    if ravdess_df.empty:
        raise RuntimeError("No valid RAVDESS audio files found in data/ravdess/")
    if cremad_df.empty:
        raise RuntimeError("No valid CREMA-D audio files found in data/cremad/")

    print(f"\n[1] Found Audio Samples:")
    print(f"    - RAVDESS (Source Domain): {len(ravdess_df)} files")
    print(f"    - CREMA-D (Target Domain): {len(cremad_df)} files")

    # 2. Extract Features
    X_rav, y_rav = extract_features_for_df(ravdess_df, desc="RAVDESS")
    X_crem, y_crem = extract_features_for_df(cremad_df, desc="CREMA-D")

    # 3. Splits
    # Source domain: 80% train, 20% test
    X_rav_train, X_rav_test, y_rav_train, y_rav_test = train_test_split(
        X_rav, y_rav, test_size=0.2, random_state=random_state, stratify=y_rav
    )

    # Target domain: 50% reference pool (unlabeled), 50% evaluation set
    X_crem_ref, X_crem_eval, y_crem_ref, y_crem_eval = train_test_split(
        X_crem, y_crem, test_size=0.5, random_state=random_state, stratify=y_crem
    )

    print(f"\n[2] Partition Statistics:")
    print(f"    - RAVDESS Train Set:       {len(X_rav_train)} samples")
    print(f"    - RAVDESS Test Set:        {len(X_rav_test)} samples")
    print(f"    - CREMA-D Reference Pool:  {len(X_crem_ref)} samples (used strictly for CORAL stats)")
    print(f"    - CREMA-D Evaluation Set:  {len(X_crem_eval)} samples (held-out test set)")

    print(f"\n[3] Class Distributions:")
    print("    - RAVDESS Train Distribution:")
    for emo, cnt in pd.Series(y_rav_train).value_counts().items():
        print(f"        * {emo:<10}: {cnt}")
    print("    - CREMA-D Evaluation Distribution:")
    for emo, cnt in pd.Series(y_crem_eval).value_counts().items():
        print(f"        * {emo:<10}: {cnt}")

    # 4. Train Random Forest Classifier
    print(f"\n[4] Training RandomForestClassifier (n_estimators=200, random_state={random_state})...")
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X_rav_train, y_rav_train)

    # 5. Evaluate Within-Domain (RAVDESS Test)
    rav_pred = clf.predict(X_rav_test)
    rav_acc = accuracy_score(y_rav_test, rav_pred)

    # 6. Fit CORAL using RAVDESS Train and CREMA-D Reference Pool
    print(f"\n[5] Fitting CORAL domain adaptation...")
    print(f"    - Estimating source stats from RAVDESS Train ({len(X_rav_train)} samples)")
    print(f"    - Estimating target stats from CREMA-D Reference Pool ({len(X_crem_ref)} samples)")
    coral = CORAL(reg=coral_reg)
    coral.fit(source_features=X_rav_train, target_features=X_crem_ref)

    # 7. Evaluate Cross-Domain (CREMA-D Evaluation Set)
    # A. Before CORAL
    crem_pred_before = clf.predict(X_crem_eval)
    crem_acc_before = accuracy_score(y_crem_eval, crem_pred_before)

    # B. After CORAL
    X_crem_eval_coral = coral.transform(X_crem_eval)
    crem_pred_after = clf.predict(X_crem_eval_coral)
    crem_acc_after = accuracy_score(y_crem_eval, crem_pred_after)

    acc_diff = crem_acc_after - crem_acc_before
    recovery_pct = (acc_diff / (rav_acc - crem_acc_before) * 100) if (rav_acc - crem_acc_before) > 0 else 0.0

    # 8. Report Results
    print("\n" + "=" * 70)
    print("EVALUATION & BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  * RAVDESS Within-Domain Test Accuracy:         {rav_acc * 100:6.2f}%")
    print(f"  * CREMA-D Cross-Domain Accuracy (BEFORE CORAL): {crem_acc_before * 100:6.2f}%")
    print(f"  * CREMA-D Cross-Domain Accuracy (AFTER CORAL):  {crem_acc_after * 100:6.2f}%")
    print(f"  * Absolute Accuracy Change:                    {acc_diff * 100:+6.2f}% points")
    print(f"  * Domain Shift Recovery:                       {recovery_pct:6.2f}%")
    print("=" * 70)

    # 9. Save Artifacts for FastAPI
    print(f"\n[6] Saving trained model and adaptation parameters...")
    joblib.dump(clf, MODEL_PATH)
    print(f"    - Saved Model to: {MODEL_PATH}")

    np.savez(
        CORAL_PATH,
        mu_source=coral.mu_source,
        mu_target=coral.mu_target,
        transform_matrix=coral.transform_matrix,
        cov_source=coral.cov_source,
        cov_target=coral.cov_target,
        classes=np.array(clf.classes_),
        reg=np.array([coral_reg]),
    )
    print(f"    - Saved CORAL Reference to: {CORAL_PATH}")

    return {
        "ravdess_total": len(ravdess_df),
        "cremad_total": len(cremad_df),
        "cremad_ref_count": len(X_crem_ref),
        "cremad_eval_count": len(X_crem_eval),
        "ravdess_accuracy": rav_acc,
        "cremad_accuracy_before": crem_acc_before,
        "cremad_accuracy_after": crem_acc_after,
        "accuracy_diff": acc_diff,
        "recovery_pct": recovery_pct,
    }


class EmotionPredictor:
    """
    Inference helper for FastAPI backend.
    Loads saved model and CORAL reference parameters to classify speech audio.
    """

    def __init__(self, model_file: Path = MODEL_PATH, coral_file: Path = CORAL_PATH):
        self.model = joblib.load(model_file)
        coral_data = np.load(coral_file, allow_pickle=True)

        self.coral = CORAL(reg=float(coral_data["reg"][0]))
        self.coral.mu_source = coral_data["mu_source"]
        self.coral.mu_target = coral_data["mu_target"]
        self.coral.transform_matrix = coral_data["transform_matrix"]
        self.coral.cov_source = coral_data["cov_source"]
        self.coral.cov_target = coral_data["cov_target"]
        self.coral.is_fitted = True

        self.classes = list(self.model.classes_)

    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray],
        use_coral: bool = True,
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict emotion for an audio file path or extracted feature vector.
        """
        if isinstance(audio_input, (str, Path)):
            feat = extract_features(audio_input)
        else:
            feat = np.asarray(audio_input, dtype=np.float32)

        feat_vector = feat.reshape(1, -1)

        if use_coral:
            feat_vector = self.coral.transform(feat_vector)

        pred_class = str(self.model.predict(feat_vector)[0])
        probs = self.model.predict_proba(feat_vector)[0]
        prob_dict = {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}
        confidence = float(np.max(probs))

        return {
            "emotion": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "coral_applied": use_coral,
        }


if __name__ == "__main__":
    train_and_evaluate_pipeline()
