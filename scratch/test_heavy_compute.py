"""Scratch test for the combined heavy compute suite.
Validates functionality, runtime, and memory overhead before updating main.py.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generated.models import create_dataset
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

def test_suite():
    print("Testing data loading...")
    dataset = create_dataset()
    records = dataset.records
    df = pd.DataFrame(records)
    print(f"Loaded DataFrame with shape: {df.shape}")

    # Prepare features
    # Convert numeric columns
    numeric_cols = [
        'demographics.age', 'anthropometry.height', 'anthropometry.weight',
        'anthropometry.bmi', 'labs.glucose', 'vitals.systolic_bp',
        'vitals.diastolic_bp', 'vitals.pulse_rate'
    ]
    binary_cols = [
        'history.hypertensive', 'history.family_diabetes',
        'history.cardiovascular_disease', 'history.family_hypertension', 'history.stroke'
    ]

    for col in numeric_cols + binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode gender and outcome
    df['gender_male'] = (df['demographics.gender'].astype(str).str.lower() == 'male').astype(int)
    df['is_diabetic'] = (df['outcome.diabetic'].astype(str).str.lower() == 'yes').astype(int)

    feature_cols = numeric_cols + binary_cols + ['gender_male']
    X = df[feature_cols].values
    y = df['is_diabetic'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature matrix X_scaled: {X_scaled.shape}")

    # 1. t-SNE & Kernel PCA
    print("\n--- Testing 1. Manifold Learning (t-SNE & Kernel PCA) ---")
    t0 = time.perf_counter()
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, max_iter=500)
    X_kpca = kpca.fit_transform(X_scaled)
    t_kpca = time.perf_counter() - t0
    print(f"Kernel PCA completed in {t_kpca:.2f}s")

    t0 = time.perf_counter()
    tsne_kwargs = {"n_components": 2, "perplexity": 30.0, "random_state": 42, "n_jobs": 2}
    try:
        tsne = TSNE(max_iter=500, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=500, **tsne_kwargs)
    X_tsne = tsne.fit_transform(X_scaled)
    t_tsne = time.perf_counter() - t0
    print(f"t-SNE completed in {t_tsne:.2f}s, KL divergence: {getattr(tsne, 'kl_divergence_', 0.0):.4f}")

    # 2. Random Forest & Permutation
    print("\n--- Testing 2. Random Forest Ensemble & Permutation ---")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=2)
    rf.fit(X_scaled, y)
    perm = permutation_importance(rf, X_scaled, y, n_repeats=5, random_state=42, n_jobs=2)
    t_rf = time.perf_counter() - t0
    print(f"RF (5-fold CV + 5x perm) completed in {t_rf:.2f}s, Mean ROC-AUC: {scores.mean():.4f}")

    # 3. High-Order Interactions & LASSO
    print("\n--- Testing 3. High-Order Interactions & LASSO Path ---")
    t0 = time.perf_counter()
    poly = PolynomialFeatures(degree=3, include_bias=False)
    # Use top 6 numeric features to keep interaction dimension well-conditioned (~83 features)
    X_sub = X_scaled[:, :6]
    X_poly = poly.fit_transform(X_sub)
    print(f"Polynomial expansion shape: {X_poly.shape}")
    lasso_cv = LassoCV(cv=5, max_iter=2000, random_state=42, n_jobs=2)
    lasso_cv.fit(X_poly, df['labs.glucose'].values)
    t_lasso = time.perf_counter() - t0
    nonzero = np.sum(lasso_cv.coef_ != 0)
    print(f"LASSO CV completed in {t_lasso:.2f}s, Non-zero features: {nonzero}/{X_poly.shape[1]}")

    # 4. 50,000 Resamples Bootstrap
    print("\n--- Testing 4. 50,000 Bootstrap Resampling ---")
    t0 = time.perf_counter()
    n_resamples = 50000
    batch_size = 10000
    n_batches = n_resamples // batch_size
    diffs = []
    diab_mask = (y == 1)
    g_diab = df.loc[diab_mask, 'labs.glucose'].values
    g_nondiab = df.loc[~diab_mask, 'labs.glucose'].values
    n_d = len(g_diab)
    n_nd = len(g_nondiab)

    rng = np.random.default_rng(42)
    for _ in range(n_batches):
        idx_d = rng.integers(0, n_d, size=(batch_size, n_d))
        idx_nd = rng.integers(0, n_nd, size=(batch_size, n_nd))
        m_d = np.mean(g_diab[idx_d], axis=1)
        m_nd = np.mean(g_nondiab[idx_nd], axis=1)
        diffs.append(m_d - m_nd)

    all_diffs = np.concatenate(diffs)
    t_boot = time.perf_counter() - t0
    ci_low, ci_high = np.percentile(all_diffs, [2.5, 97.5])
    print(f"Bootstrap 50,000 completed in {t_boot:.2f}s ({n_resamples/t_boot:.1f} iter/s), 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # 5. GMM Clustering
    print("\n--- Testing 5. Multi-Restart GMM Clustering ---")
    t0 = time.perf_counter()
    bics = {}
    for k in [2, 3, 4, 5]:
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, max_iter=100, random_state=42)
        gmm.fit(X_scaled)
        bics[k] = gmm.bic(X_scaled)
    t_gmm = time.perf_counter() - t0
    best_k = min(bics, key=bics.get)
    print(f"GMM sweep K=2..5 completed in {t_gmm:.2f}s, Optimal K: {best_k}")

    # 6. Deep Autoencoder
    print("\n--- Testing 6. Deep Autoencoder MLP ---")
    t0 = time.perf_counter()
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 16, 4, 16, 64),
        activation='relu',
        solver='adam',
        max_iter=100,
        batch_size=64,
        random_state=42
    )
    mlp.fit(X_scaled, X_scaled)
    t_mlp = time.perf_counter() - t0
    print(f"Autoencoder completed in {t_mlp:.2f}s, Loss: {mlp.loss_:.5f}")

    total_time = t_kpca + t_tsne + t_rf + t_lasso + t_boot + t_gmm + t_mlp
    print(f"\n>>> Total compute time for all 6 jobs: {total_time:.2f}s <<<")

if __name__ == "__main__":
    test_suite()
