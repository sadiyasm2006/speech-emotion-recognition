"""
CORAL (Correlation Alignment) Domain Adaptation Module.

Implements covariance alignment to minimize the domain shift between
source domain (e.g. RAVDESS) and target domain (e.g. CREMA-D) speech features.

Reference:
    Sun, B., Feng, J., & Saenko, K. (2016).
    Return of Frustratingly Easy Domain Adaptation. AAAI.
"""

from typing import Optional, Tuple, Union
import numpy as np


class CORAL:
    """
    CORAL: Correlation Alignment for Unsupervised Domain Adaptation.

    Aligns the second-order statistics (covariance) and first-order statistics (mean)
    of the target domain features with the source domain features.
    """

    def __init__(self, reg: float = 1e-4):
        """
        Args:
            reg (float): Regularization parameter added to the covariance diagonals
                         to ensure numerical stability and positive-definiteness.
        """
        self.reg = reg
        self.mu_source: Optional[np.ndarray] = None
        self.mu_target: Optional[np.ndarray] = None
        self.cov_source: Optional[np.ndarray] = None
        self.cov_target: Optional[np.ndarray] = None
        self.transform_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    @staticmethod
    def _matrix_power_sym(mat: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
        """
        Computes mat^power for a symmetric positive semi-definite matrix
        using eigendecomposition (eigh).
        """
        evals, evecs = np.linalg.eigh(mat)
        # Clip eigenvalues for numerical stability
        evals = np.maximum(evals, eps)
        evals_pow = np.power(evals, power)
        return evecs @ np.diag(evals_pow) @ evecs.T

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "CORAL":
        """
        Learns the whitening and coloring transformation from source and target features.

        Args:
            source_features (np.ndarray): Source domain feature matrix, shape (n_source, n_features).
            target_features (np.ndarray): Target domain feature matrix, shape (n_target, n_features).

        Returns:
            self: The fitted CORAL instance.
        """
        Xs = np.asarray(source_features, dtype=np.float64)
        Xt = np.asarray(target_features, dtype=np.float64)

        if Xs.ndim != 2 or Xt.ndim != 2:
            raise ValueError(
                f"Feature matrices must be 2-dimensional. Got Xs: {Xs.shape}, Xt: {Xt.shape}"
            )

        if Xs.shape[1] != Xt.shape[1]:
            raise ValueError(
                f"Source and target feature dimensions must match. Got Xs: {Xs.shape[1]}, Xt: {Xt.shape[1]}"
            )

        n_features = Xs.shape[1]

        # 1. First-order statistics: Means
        self.mu_source = np.mean(Xs, axis=0)
        self.mu_target = np.mean(Xt, axis=0)

        # Center both distributions
        Xs_centered = Xs - self.mu_source
        Xt_centered = Xt - self.mu_target

        # 2. Second-order statistics: Covariance matrices + regularizer
        # cov = (X.T @ X) / (n - 1) + reg * I
        eye = np.eye(n_features, dtype=np.float64)
        self.cov_source = np.cov(Xs_centered, rowvar=False) + self.reg * eye
        self.cov_target = np.cov(Xt_centered, rowvar=False) + self.reg * eye

        # 3. Eigendecomposition to compute whitening and coloring operators
        # Whitening: Ct^(-1/2)
        Ct_inv_sqrt = self._matrix_power_sym(self.cov_target, power=-0.5)

        # Coloring: Cs^(1/2)
        Cs_sqrt = self._matrix_power_sym(self.cov_source, power=0.5)

        # 4. Composite CORAL transform matrix: A = Ct^(-1/2) * Cs^(1/2)
        self.transform_matrix = Ct_inv_sqrt @ Cs_sqrt
        self.is_fitted = True
        return self

    def transform(self, target_features: np.ndarray) -> np.ndarray:
        """
        Applies the CORAL domain adaptation transformation to target features.

        Args:
            target_features (np.ndarray): Target features, shape (n_samples, n_features) or (n_features,).

        Returns:
            np.ndarray: Aligned target features matching the source domain space.
        """
        if not self.is_fitted or self.transform_matrix is None:
            raise RuntimeError("CORAL instance must be fitted before calling transform().")

        Xt = np.asarray(target_features, dtype=np.float64)
        is_single_sample = Xt.ndim == 1

        if is_single_sample:
            Xt = Xt[np.newaxis, :]

        if Xt.shape[1] != self.transform_matrix.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.transform_matrix.shape[0]}, got {Xt.shape[1]}"
            )

        # Apply transformation: (Xt - mu_target) @ A + mu_source
        Xt_centered = Xt - self.mu_target
        Xt_aligned = Xt_centered @ self.transform_matrix + self.mu_source

        if is_single_sample:
            return Xt_aligned[0].astype(target_features.dtype if hasattr(target_features, 'dtype') else np.float32)

        return Xt_aligned.astype(target_features.dtype if hasattr(target_features, 'dtype') else np.float32)

    def fit_transform(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> np.ndarray:
        """
        Fits CORAL and returns the aligned target features.
        """
        self.fit(source_features, target_features)
        return self.transform(target_features)


def coral_align(
    source_features: np.ndarray,
    target_features: np.ndarray,
    reg: float = 1e-4,
) -> Tuple[np.ndarray, CORAL]:
    """
    Functional interface for CORAL domain adaptation.

    Returns:
        aligned_target (np.ndarray): Transformed target domain features.
        coral_model (CORAL): Fitted CORAL transformer instance.
    """
    coral = CORAL(reg=reg)
    aligned_target = coral.fit_transform(source_features, target_features)
    return aligned_target, coral


def frobenius_covariance_distance(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """
    Computes the Frobenius norm distance ||Cov_A - Cov_B||_F between two covariance matrices.
    """
    return float(np.linalg.norm(cov_a - cov_b, ord="fro"))


def test_coral():
    """
    Self-contained verification test for CORAL domain adaptation.
    """
    print("=" * 65)
    print("CORAL DOMAIN ADAPTATION VERIFICATION TEST")
    print("=" * 65)

    np.random.seed(42)
    n_source, n_target, n_features = 200, 180, 80

    # 1. Generate synthetic source & target feature matrices with significant domain shift
    print(f"Generating synthetic feature matrices (dim = {n_features}):")
    print(f"  * Source features shape: ({n_source}, {n_features}) [mean=2.5, scale=1.8]")
    print(f"  * Target features shape: ({n_target}, {n_features}) [mean=-1.5, scale=0.6]")

    source_features = np.random.randn(n_source, n_features) * 1.8 + 2.5
    target_features = np.random.randn(n_target, n_features) * 0.6 - 1.5

    # 2. Compute initial statistics & covariance distance before CORAL
    cov_source = np.cov(source_features, rowvar=False)
    cov_target_before = np.cov(target_features, rowvar=False)
    dist_before = frobenius_covariance_distance(cov_source, cov_target_before)

    print(f"\nBefore Adaptation:")
    print(f"  * Source feature mean:     {np.mean(source_features):.4f}")
    print(f"  * Target feature mean:     {np.mean(target_features):.4f}")
    print(f"  * Covariance Frobenius dist: {dist_before:.4f}")

    # 3. Fit and apply CORAL
    coral = CORAL(reg=1e-4)
    aligned_target = coral.fit_transform(source_features, target_features)

    # 4. Verify dimensions
    print(f"\nAfter CORAL Adaptation:")
    print(f"  * Aligned target shape:    {aligned_target.shape}")
    assert aligned_target.shape == target_features.shape, "Shape mismatch in output!"

    # 5. Compute statistics & distance after CORAL
    cov_target_after = np.cov(aligned_target, rowvar=False)
    dist_after = frobenius_covariance_distance(cov_source, cov_target_after)
    distance_reduction = (1 - dist_after / dist_before) * 100

    print(f"  * Aligned target mean:     {np.mean(aligned_target):.4f} (Source mean: {np.mean(source_features):.4f})")
    print(f"  * Covariance Frobenius dist: {dist_after:.4f}")
    print(f"  * Covariance alignment reduction: {distance_reduction:.2f}% closer to source")

    # 6. Test single sample transformation
    single_sample = target_features[0]
    single_aligned = coral.transform(single_sample)
    print(f"  * Single sample shape:     {single_aligned.shape} (Expected: ({n_features},))")
    assert single_aligned.shape == (n_features,), "Single sample transform shape error!"

    print("\nVerification Test Result: PASSED (All checks successful)")
    print("=" * 65)


if __name__ == "__main__":
    test_coral()
