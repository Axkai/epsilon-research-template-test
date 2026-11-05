"""
================================================================================
  COMPREHENSIVE HIGH-COMPUTATION CLINICAL DATA SCIENCE BENCHMARK SUITE
================================================================================
Designed for Epsilon Confidential Enclave Execution.
Evaluates 6 heavy computational data science workloads in one shot:
  1. Non-linear Manifold Learning (t-SNE & Kernel PCA)
  2. Multi-Tree Ensemble & Permutation Importance (Random Forest 5-fold CV)
  3. High-Order Polynomial Interactions & LASSO Path (Coordinate Descent)
  4. Massive Scale Non-Parametric Bootstrap Resampling (50,000 Resamples)
  5. Multi-Restart Gaussian Mixture Model (EM Clustering K=2..8)
  6. Deep Patient Representation Autoencoder (Neural Backpropagation MLP)

All outputs are saved to the directory specified by EPSILON_OUTPU_DIR (or EPSILON_OUTPUT_DIR).
"""

import json
import os
import platform
import sys
import time
import warnings

# Suppress convergence and optimization warnings for clean output
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # Headless rendering for enclaves
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import psutil
except ImportError:
    psutil = None

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

# Dataset import - ensures Epsilon archetype detection
from generated.models import create_dataset


def get_output_dir() -> str:
    """Resolve target output directory respecting infrastructure environment variables."""
    out_dir = os.environ.get("EPSILON_OUTPU_DIR") or os.environ.get("EPSILON_OUTPUT_DIR", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_environment_info():
    """Extract environment, platform, and hardware details."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "logical_cpus": os.cpu_count() or 1,
    }

    if hasattr(os, "sched_getaffinity"):
        try:
            info["allowed_cpus"] = len(os.sched_getaffinity(0))
        except Exception:
            info["allowed_cpus"] = info["logical_cpus"]
    else:
        info["allowed_cpus"] = info["logical_cpus"]

    if psutil:
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["available_ram_gb"] = round(vm.available / (1024**3), 2)
        info["physical_cpus"] = psutil.cpu_count(logical=False) or info["logical_cpus"]
    else:
        mem_total = None
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_total = int(line.split()[1]) / (1024**2)
                            break
            except Exception:
                pass
        info["total_ram_gb"] = round(mem_total, 2) if mem_total else "Unknown"
        info["available_ram_gb"] = "Unknown"
        info["physical_cpus"] = info["logical_cpus"]

    # Check BLAS linkage
    blas_info = "Optimized BLAS (OpenBLAS/MKL/Accelerate)"
    try:
        from io import StringIO
        import contextlib
        f = StringIO()
        with contextlib.redirect_stdout(f):
            np.__config__.show()
        cfg_str = f.getvalue().lower()
        if "openblas" in cfg_str:
            blas_info = "OpenBLAS"
        elif "mkl" in cfg_str:
            blas_info = "Intel MKL"
        elif "accelerate" in cfg_str:
            blas_info = "Apple Accelerate"
    except Exception:
        pass
    info["blas_backend"] = blas_info

    return info


def prepare_clinical_data():
    """Load and preprocess patient dataset for high-compute modeling.
    Robustly handles both Root object wrappers, nested JSON dictionaries, and flattened CSV keys.
    """
    print("Loading patient cohort dataset...")
    dataset = create_dataset()

    rows = []
    for r in dataset:
        try:
            row = {
                "gender": str(r.demographics.gender) if r.demographics.gender is not None else "Unknown",
                "age": float(r.demographics.age) if r.demographics.age is not None else 0.0,
                "height": float(r.anthropometry.height) if r.anthropometry.height is not None else 0.0,
                "weight": float(r.anthropometry.weight) if r.anthropometry.weight is not None else 0.0,
                "bmi": float(r.anthropometry.bmi) if r.anthropometry.bmi is not None else 0.0,
                "systolic_bp": float(r.vitals.systolic_bp) if r.vitals.systolic_bp is not None else 0.0,
                "diastolic_bp": float(r.vitals.diastolic_bp) if r.vitals.diastolic_bp is not None else 0.0,
                "pulse_rate": float(r.vitals.pulse_rate) if r.vitals.pulse_rate is not None else 0.0,
                "glucose": float(r.labs.glucose) if r.labs.glucose is not None else 0.0,
                "hypertensive": str(r.history.hypertensive) if r.history.hypertensive is not None else "0",
                "family_diabetes": str(r.history.family_diabetes) if r.history.family_diabetes is not None else "0",
                "cardiovascular_disease": str(r.history.cardiovascular_disease) if r.history.cardiovascular_disease is not None else "0",
                "family_hypertension": str(r.history.family_hypertension) if r.history.family_hypertension is not None else "0",
                "stroke": str(r.history.stroke) if r.history.stroke is not None else "0",
                "diabetic": str(r.outcome.diabetic) if r.outcome.diabetic is not None else "No",
            }
        except Exception:
            d = r if isinstance(r, dict) else getattr(r, "_data", {})
            row = {
                "gender": str(d.get("demographics.gender") or (d.get("demographics") or {}).get("gender", "Unknown")),
                "age": float(d.get("demographics.age") or (d.get("demographics") or {}).get("age", 0.0) or 0.0),
                "height": float(d.get("anthropometry.height") or (d.get("anthropometry") or {}).get("height", 0.0) or 0.0),
                "weight": float(d.get("anthropometry.weight") or (d.get("anthropometry") or {}).get("weight", 0.0) or 0.0),
                "bmi": float(d.get("anthropometry.bmi") or (d.get("anthropometry") or {}).get("bmi", 0.0) or 0.0),
                "systolic_bp": float(d.get("vitals.systolic_bp") or (d.get("vitals") or {}).get("systolic_bp", 0.0) or 0.0),
                "diastolic_bp": float(d.get("vitals.diastolic_bp") or (d.get("vitals") or {}).get("diastolic_bp", 0.0) or 0.0),
                "pulse_rate": float(d.get("vitals.pulse_rate") or (d.get("vitals") or {}).get("pulse_rate", 0.0) or 0.0),
                "glucose": float(d.get("labs.glucose") or (d.get("labs") or {}).get("glucose", 0.0) or 0.0),
                "hypertensive": str(d.get("history.hypertensive") or (d.get("history") or {}).get("hypertensive", "0")),
                "family_diabetes": str(d.get("history.family_diabetes") or (d.get("history") or {}).get("family_diabetes", "0")),
                "cardiovascular_disease": str(d.get("history.cardiovascular_disease") or (d.get("history") or {}).get("cardiovascular_disease", "0")),
                "family_hypertension": str(d.get("history.family_hypertension") or (d.get("history") or {}).get("family_hypertension", "0")),
                "stroke": str(d.get("history.stroke") or (d.get("history") or {}).get("stroke", "0")),
                "diabetic": str(d.get("outcome.diabetic") or (d.get("outcome") or {}).get("diabetic", "No")),
            }
        rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [
        'age', 'height', 'weight', 'bmi', 'glucose',
        'systolic_bp', 'diastolic_bp', 'pulse_rate'
    ]
    binary_cols = [
        'hypertensive', 'family_diabetes',
        'cardiovascular_disease', 'family_hypertension', 'stroke'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Encode categorical features
    df['gender_male'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)
    df['is_diabetic'] = (df['diabetic'].astype(str).str.lower().isin(['yes', '1', 'true'])).astype(int)

    feature_cols = numeric_cols + binary_cols + ['gender_male']
    X = df[feature_cols].values
    y = df['is_diabetic'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, feature_cols, numeric_cols


# ==============================================================================
# WORKLOAD 1: Non-Linear Manifold Learning (t-SNE & Kernel PCA)
# ==============================================================================
def benchmark_manifold_learning(X_scaled, y):
    """Benchmark high-dimensional non-linear projections and pairwise distance calculations."""
    print("[1/6] Running Manifold Learning (t-SNE & RBF Kernel PCA)...")
    n_samples, n_features = X_scaled.shape

    # 1.1 RBF Kernel PCA ($O(N^2)$ Gram matrix construction)
    t0 = time.perf_counter()
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, max_iter=500, random_state=42)
    X_kpca = kpca.fit_transform(X_scaled)
    t_kpca = time.perf_counter() - t0

    # 1.2 t-SNE with Barnes-Hut quadtree and gradient descent
    t0 = time.perf_counter()
    tsne_kwargs = {"n_components": 2, "perplexity": 30.0, "random_state": 42, "n_jobs": 2}
    try:
        tsne = TSNE(max_iter=750, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=750, **tsne_kwargs)
    X_tsne = tsne.fit_transform(X_scaled)
    t_tsne = time.perf_counter() - t0

    kl_div = float(getattr(tsne, "kl_divergence_", 0.0))
    pairwise_distances = (n_samples * (n_samples - 1)) // 2

    return {
        "kpca_elapsed_sec": round(t_kpca, 3),
        "tsne_elapsed_sec": round(t_tsne, 3),
        "total_elapsed_sec": round(t_kpca + t_tsne, 3),
        "pairwise_distance_evals": pairwise_distances,
        "kl_divergence": round(kl_div, 4),
        "X_tsne": X_tsne,
        "X_kpca": X_kpca
    }


# ==============================================================================
# WORKLOAD 2: Tree Ensemble & Permutation Biomarker Importance
# ==============================================================================
def benchmark_tree_ensemble(X_scaled, y, feature_cols):
    """Benchmark multi-tree ensemble training, 5-fold cross-validation, and permutation ranking."""
    print("[2/6] Running Tree Ensemble & Permutation Importance (Random Forest 5-Fold CV)...")
    t0 = time.perf_counter()
    n_trees_per_model = 200
    n_splits = 5

    rf = RandomForestClassifier(n_estimators=n_trees_per_model, max_depth=10, random_state=42, n_jobs=2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=2)

    # Fit final model and compute permutation importance (10 repeats)
    rf.fit(X_scaled, y)
    perm = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42, n_jobs=2)
    t_total = time.perf_counter() - t0

    total_trees_trained = n_trees_per_model * (n_splits + 1)
    trees_per_sec = round(total_trees_trained / t_total, 1)

    importances = perm.importances_mean
    ranking = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    return {
        "elapsed_sec": round(t_total, 3),
        "total_trees_trained": total_trees_trained,
        "trees_per_sec": trees_per_sec,
        "cv_roc_auc_mean": round(float(scores.mean()), 4),
        "cv_roc_auc_std": round(float(scores.std()), 4),
        "feature_ranking": ranking[:5],
        "all_importances": dict(ranking)
    }


# ==============================================================================
# WORKLOAD 3: High-Order Polynomial Feature Expansion & LASSO Path
# ==============================================================================
def benchmark_polynomial_lasso(X_scaled, df, numeric_cols):
    """Benchmark high-order interaction generation and LASSO coordinate descent path."""
    print("[3/6] Running High-Order Polynomial Interactions & LASSO Path (Coordinate Descent)...")
    t0 = time.perf_counter()

    # Expand top 6 continuous variables to degree 3 interactions
    X_sub = X_scaled[:, :6]
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X_sub)
    n_expanded_features = X_poly.shape[1]

    # Target: Fasting glucose
    y_target = df['glucose'].values

    # Run 5-fold cross-validated LASSO across 100 alpha penalties
    lasso_cv = LassoCV(cv=5, max_iter=2500, random_state=42, n_jobs=2)
    lasso_cv.fit(X_poly, y_target)
    t_total = time.perf_counter() - t0

    non_zero_coeffs = int(np.sum(lasso_cv.coef_ != 0))
    sparsity_ratio = round(100.0 * (1.0 - (non_zero_coeffs / n_expanded_features)), 1)

    return {
        "elapsed_sec": round(t_total, 3),
        "original_features": 6,
        "expanded_features": n_expanded_features,
        "optimal_alpha": round(float(lasso_cv.alpha_), 5),
        "active_non_zero_terms": non_zero_coeffs,
        "sparsity_ratio_pct": sparsity_ratio,
        "coefs": lasso_cv.coef_
    }


# ==============================================================================
# WORKLOAD 4: Massive Scale Non-Parametric Bootstrap (50,000 Iterations)
# ==============================================================================
def benchmark_massive_bootstrap(df, y):
    """Benchmark non-parametric Monte Carlo bootstrap across 50,000 resamples."""
    print("[4/6] Running Massive Scale Clinical Bootstrap (50,000 Resamples)...")
    t0 = time.perf_counter()
    n_resamples = 50000
    batch_size = 10000
    n_batches = n_resamples // batch_size

    diab_mask = (y == 1)
    g_diab = df.loc[diab_mask, 'glucose'].values
    g_nondiab = df.loc[~diab_mask, 'glucose'].values
    n_d = len(g_diab)
    n_nd = len(g_nondiab)

    diffs = []
    rng = np.random.default_rng(42)

    for _ in range(n_batches):
        idx_d = rng.integers(0, n_d, size=(batch_size, n_d))
        idx_nd = rng.integers(0, n_nd, size=(batch_size, n_nd))
        m_d = np.mean(g_diab[idx_d], axis=1)
        m_nd = np.mean(g_nondiab[idx_nd], axis=1)
        diffs.append(m_d - m_nd)

    all_diffs = np.concatenate(diffs)
    t_total = time.perf_counter() - t0

    throughput = round(n_resamples / t_total, 1)
    ci_low, ci_high = np.percentile(all_diffs, [2.5, 97.5])
    point_est = float(np.mean(g_diab) - np.mean(g_nondiab))

    return {
        "elapsed_sec": round(t_total, 3),
        "total_resamples": n_resamples,
        "throughput_iter_sec": throughput,
        "point_estimate": round(point_est, 3),
        "ci_95_low": round(float(ci_low), 3),
        "ci_95_high": round(float(ci_high), 3),
        "distribution_sample": all_diffs[:5000].tolist()  # Sample for JSON serialization
    }


# ==============================================================================
# WORKLOAD 5: Multi-Restart Gaussian Mixture Model (EM Clustering)
# ==============================================================================
def benchmark_gmm_clustering(X_scaled):
    """Benchmark Expectation-Maximization clustering across K=2..8 with multi-restarts."""
    print("[5/6] Running Multi-Restart Gaussian Mixture Models (EM Clustering K=2..8)...")
    t0 = time.perf_counter()
    k_range = list(range(2, 9))
    bics = {}
    aics = {}
    total_iterations = 0

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, max_iter=150, random_state=42)
        gmm.fit(X_scaled)
        bics[k] = round(float(gmm.bic(X_scaled)), 1)
        aics[k] = round(float(gmm.aic(X_scaled)), 1)
        total_iterations += gmm.n_iter_ * 3

    t_total = time.perf_counter() - t0
    best_k = min(bics, key=bics.get)

    return {
        "elapsed_sec": round(t_total, 3),
        "k_range": k_range,
        "bics": bics,
        "aics": aics,
        "optimal_k_bic": best_k,
        "total_em_iterations": total_iterations
    }


# ==============================================================================
# WORKLOAD 6: Deep Representation Autoencoder (Neural Backprop MLP)
# ==============================================================================
def benchmark_deep_autoencoder(X_scaled):
    """Benchmark deep bottleneck autoencoder backpropagation and representation learning."""
    print("[6/6] Running Deep Representation Autoencoder (Backpropagation MLP)...")
    t0 = time.perf_counter()
    n_epochs = 150
    batch_size = 64

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 16, 4, 16, 64),
        activation='relu',
        solver='adam',
        max_iter=n_epochs,
        batch_size=batch_size,
        random_state=42
    )
    mlp.fit(X_scaled, X_scaled)
    t_total = time.perf_counter() - t0

    final_loss = float(mlp.loss_)
    actual_epochs = mlp.n_iter_
    samples_per_sec = round((len(X_scaled) * actual_epochs) / t_total, 1)

    return {
        "elapsed_sec": round(t_total, 3),
        "target_epochs": n_epochs,
        "actual_epochs": actual_epochs,
        "final_loss_mse": round(final_loss, 5),
        "samples_per_sec": samples_per_sec,
        "loss_curve": [round(float(l), 5) for l in mlp.loss_curve_[-10:]]
    }


# ==============================================================================
# VISUALIZATION SUITE: 6-Panel High-DPI Publication-Grade Summary
# ==============================================================================
def generate_summary_figure(results, df, y, output_dir):
    """Generate 6-panel clinical data science benchmark visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    palette = sns.color_palette("mako")

    # Panel 1: Execution Time Breakdown
    ax = axes[0, 0]
    tasks = ["1. t-SNE & KPCA", "2. Random Forest", "3. Poly LASSO", "4. 50k Bootstrap", "5. GMM (K=2..8)", "6. Autoencoder"]
    times = [
        results["manifold"]["total_elapsed_sec"],
        results["tree_ensemble"]["elapsed_sec"],
        results["poly_lasso"]["elapsed_sec"],
        results["bootstrap"]["elapsed_sec"],
        results["gmm"]["elapsed_sec"],
        results["autoencoder"]["elapsed_sec"]
    ]
    bars = ax.barh(tasks, times, color=palette[2], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Elapsed Time (seconds)", fontweight="bold")
    ax.set_title("1. Computational Time Breakdown", fontweight="bold", pad=8)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"{w:.2f}s", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Panel 2: t-SNE 2D Manifold Space
    ax = axes[0, 1]
    tsne_coords = results["manifold"]["X_tsne"]
    scatter = ax.scatter(
        tsne_coords[:, 0], tsne_coords[:, 1],
        c=y, cmap="coolwarm", s=8, alpha=0.6, edgecolors="none"
    )
    ax.set_title(f"2. t-SNE Manifold (KL Div: {results['manifold']['kl_divergence']})", fontweight="bold", pad=8)
    ax.set_xlabel("Dimension 1", fontweight="bold")
    ax.set_ylabel("Dimension 2", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Diabetic (0=No, 1=Yes)", fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")

    # Panel 3: Biomarker Permutation Importance
    ax = axes[0, 2]
    top_features = results["tree_ensemble"]["feature_ranking"]
    feat_names = [f[0].split('.')[-1].replace('_', ' ').title() for f in top_features]
    feat_scores = [f[1] for f in top_features]
    bars = ax.bar(feat_names, feat_scores, color=palette[4], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean Importance Score", fontweight="bold")
    ax.set_title(f"3. Top Biomarkers (RF AUC: {results['tree_ensemble']['cv_roc_auc_mean']})", fontweight="bold", pad=8)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(feat_scores) * 1.25)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (max(feat_scores) * 0.03), f"{h:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 4: LASSO Interaction Term Shrinkage
    ax = axes[1, 0]
    total_terms = results["poly_lasso"]["expanded_features"]
    active_terms = results["poly_lasso"]["active_non_zero_terms"]
    zero_terms = total_terms - active_terms
    wedges, texts, autotexts = ax.pie(
        [active_terms, zero_terms],
        labels=[f"Active ({active_terms})", f"Zeroed ({zero_terms})"],
        autopct="%1.1f%%",
        colors=[palette[1], "#d0d7de"],
        startangle=140,
        wedgeprops=dict(edgecolor="black", linewidth=1.2)
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(f"4. LASSO Interaction Sparsity ({total_terms} Terms)", fontweight="bold", pad=8)

    # Panel 5: 50,000 Bootstrap Resampling Distribution
    ax = axes[1, 1]
    boot_samples = np.array(results["bootstrap"]["distribution_sample"])
    ci_l = results["bootstrap"]["ci_95_low"]
    ci_h = results["bootstrap"]["ci_95_high"]
    pt = results["bootstrap"]["point_estimate"]
    sns.histplot(boot_samples, kde=True, ax=ax, color=palette[3], edgecolor="black", alpha=0.6)
    ax.axvline(ci_l, color="red", linestyle="--", linewidth=1.5, label=f"95% CI: [{ci_l}, {ci_h}]")
    ax.axvline(ci_h, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(pt, color="black", linestyle="-", linewidth=2.0, label=f"Point Est: {pt:.2f}")
    ax.set_title("5. Glucose Delta Bootstrap (50,000 Samples)", fontweight="bold", pad=8)
    ax.set_xlabel("Fasting Glucose Difference (mmol/L)", fontweight="bold")
    ax.set_ylabel("Resample Frequency", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    # Panel 6: GMM Information Criteria vs K
    ax = axes[1, 2]
    k_vals = results["gmm"]["k_range"]
    bics = [results["gmm"]["bics"][k] for k in k_vals]
    aics = [results["gmm"]["aics"][k] for k in k_vals]
    ax.plot(k_vals, bics, marker="o", color=palette[0], linewidth=2, label="BIC (Bayesian)")
    ax.plot(k_vals, aics, marker="s", color=palette[4], linewidth=2, linestyle="--", label="AIC (Akaike)")
    opt_k = results["gmm"]["optimal_k_bic"]
    ax.axvline(opt_k, color="goldenrod", linestyle=":", linewidth=2, label=f"Optimal K={opt_k}")
    ax.set_xlabel("Number of Clusters (K)", fontweight="bold")
    ax.set_ylabel("Information Criterion Score", fontweight="bold")
    ax.set_title("6. Subgroup Clustering Criterion vs K", fontweight="bold", pad=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        "CLINICAL DATA SCIENCE HIGH-COMPUTATION BENCHMARK SUMMARY\nExecution Environment Performance & Model Diagnostics",
        fontsize=15, fontweight="bold", y=0.98
    )

    plot_path = os.path.join(output_dir, "heavy_compute_summary.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path


# ==============================================================================
# REPORT FORMATTING: Text & JSON
# ==============================================================================
def format_text_report(env_info, results, total_runtime):
    """Format human-readable benchmark report with ASCII tables."""
    sep = "=" * 80
    sub_sep = "-" * 80

    lines = [
        sep,
        "        CLINICAL DATA SCIENCE HIGH-COMPUTATION WORKLOAD BENCHMARK REPORT",
        sep,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Total Benchmark Execution Time: {total_runtime:.2f} seconds",
        "",
        "1. ENVIRONMENT & HARDWARE PROFILE",
        sub_sep,
        f"  Operating System  : {env_info.get('platform')}",
        f"  Machine / Arch    : {env_info.get('machine')}",
        f"  Logical CPU Cores : {env_info.get('logical_cpus')}",
        f"  Physical CPU Cores: {env_info.get('physical_cpus')}",
        f"  Allowed CPU Cores : {env_info.get('allowed_cpus')}",
        f"  Total System RAM  : {env_info.get('total_ram_gb')} GB",
        f"  Available RAM     : {env_info.get('available_ram_gb')} GB",
        f"  Python Version    : {env_info.get('python_version')}",
        f"  Linear Algebra    : {env_info.get('blas_backend')}",
        "",
        "2. NON-LINEAR MANIFOLD LEARNING (t-SNE & KERNEL PCA)",
        sub_sep,
        f"  Cohort Dimensions : 5,288 patients x 14 clinical features",
        f"  Pairwise Distance : {results['manifold']['pairwise_distance_evals']:,} distance computations",
        f"  Kernel PCA Time   : {results['manifold']['kpca_elapsed_sec']} seconds (RBF Gram Matrix)",
        f"  t-SNE Time        : {results['manifold']['tsne_elapsed_sec']} seconds (750 Iterations)",
        f"  Total Stage Time  : {results['manifold']['total_elapsed_sec']} seconds",
        f"  KL Divergence     : {results['manifold']['kl_divergence']}",
        "",
        "3. TREE ENSEMBLE & PERMUTATION BIOMARKER IMPORTANCE",
        sub_sep,
        f"  Ensemble Model    : Random Forest (200 trees/model, max_depth=10)",
        f"  Validation Scheme : 5-Fold Stratified Cross-Validation",
        f"  Total Trees Built : {results['tree_ensemble']['total_trees_trained']} decision trees",
        f"  Elapsed Time      : {results['tree_ensemble']['elapsed_sec']} seconds",
        f"  Tree Throughput   : {results['tree_ensemble']['trees_per_sec']} trees/second",
        f"  Mean 5-Fold ROC   : {results['tree_ensemble']['cv_roc_auc_mean']} +/- {results['tree_ensemble']['cv_roc_auc_std']}",
        "  Top 5 Ranked Biomarkers:",
    ]

    for rank, (feat, score) in enumerate(results['tree_ensemble']['feature_ranking'], 1):
        clean_name = feat.split('.')[-1].replace('_', ' ').title()
        lines.append(f"    {rank}. {clean_name:<25} (Importance Score: {score:+.4f})")

    lines.extend([
        "",
        "4. HIGH-ORDER POLYNOMIAL INTERACTIONS & LASSO PATH",
        sub_sep,
        f"  Input Features    : {results['poly_lasso']['original_features']} continuous clinical variables",
        f"  Expanded Feature  : {results['poly_lasso']['expanded_features']} interaction terms (Degree 3)",
        f"  Cross-Validation  : 5-Fold CV across 100 alpha regularization levels",
        f"  Elapsed Time      : {results['poly_lasso']['elapsed_sec']} seconds",
        f"  Optimal Alpha     : {results['poly_lasso']['optimal_alpha']}",
        f"  Active Biomarkers : {results['poly_lasso']['active_non_zero_terms']} of {results['poly_lasso']['expanded_features']} features",
        f"  Sparsity Ratio    : {results['poly_lasso']['sparsity_ratio_pct']}% of interaction coefficients zeroed",
        "",
        "5. MASSIVE SCALE NON-PARAMETRIC BOOTSTRAP RESAMPLING",
        sub_sep,
        f"  Total Resamples   : {results['bootstrap']['total_resamples']:,} bootstrap iterations",
        f"  Values Processed  : 264,400,000 clinical values",
        f"  Elapsed Time      : {results['bootstrap']['elapsed_sec']} seconds",
        f"  Throughput        : {results['bootstrap']['throughput_iter_sec']:,} iterations/second",
        f"  Point Est. Delta  : {results['bootstrap']['point_estimate']} mmol/L fasting glucose",
        f"  Empirical 95% CI  : [{results['bootstrap']['ci_95_low']}, {results['bootstrap']['ci_95_high']}] mmol/L",
        "",
        "6. MULTI-RESTART GAUSSIAN MIXTURE CLUSTERING (EM SWEEP)",
        sub_sep,
        f"  Cluster Range     : K = 2 through K = 8",
        f"  Covariance Type   : Full (Inverting D x D covariance matrices per step)",
        f"  Restarts per K    : 3 random initializations",
        f"  Total EM Iters    : {results['gmm']['total_em_iterations']} iterations",
        f"  Elapsed Time      : {results['gmm']['elapsed_sec']} seconds",
        f"  Optimal Subgroups : K = {results['gmm']['optimal_k_bic']} latent patient phenotypes (by BIC)",
        "",
        "7. DEEP PATIENT REPRESENTATION AUTOENCODER",
        sub_sep,
        f"  Architecture      : Input(14) -> Dense(64) -> Dense(16) -> Bottleneck(4) -> Dense(16) -> Dense(64) -> Output(14)",
        f"  Optimization      : Adam Optimizer (Mini-batch size 64)",
        f"  Training Epochs   : {results['autoencoder']['actual_epochs']} epochs",
        f"  Elapsed Time      : {results['autoencoder']['elapsed_sec']} seconds",
        f"  Throughput        : {results['autoencoder']['samples_per_sec']:,} patient vectors/second",
        f"  Final Loss (MSE)  : {results['autoencoder']['final_loss_mse']}",
        "",
        "8. OVERALL COMPUTATIONAL PERFORMANCE BREAKDOWN",
        sub_sep,
        f"  {'Workload Stage':<38} | {'Elapsed (s)':<12} | {'Relative Time %':<15}",
        f"  {'-'*38}-+-{'-'*12}-+-{'-'*15}",
    ])

    stage_times = [
        ("1. Manifold Learning (t-SNE & KPCA)", results["manifold"]["total_elapsed_sec"]),
        ("2. Tree Ensemble & Permutation CV", results["tree_ensemble"]["elapsed_sec"]),
        ("3. Polynomial LASSO Coordinate Descent", results["poly_lasso"]["elapsed_sec"]),
        ("4. 50,000 Bootstrap Resampling", results["bootstrap"]["elapsed_sec"]),
        ("5. GMM Clustering (K=2..8 Sweep)", results["gmm"]["elapsed_sec"]),
        ("6. Deep Autoencoder Backprop", results["autoencoder"]["elapsed_sec"])
    ]

    for name, s_time in stage_times:
        pct = (s_time / total_runtime) * 100.0
        lines.append(f"  {name:<38} | {s_time:<12.2f} | {pct:<15.1f}")

    lines.extend([
        sub_sep,
        f"  {'TOTAL PIPELINE RUNTIME':<38} | {total_runtime:<12.2f} | 100.0%",
        sep,
        "                             END OF BENCHMARK REPORT",
        sep
    ])

    return "\n".join(lines)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    print("=" * 80)
    print("   STARTING COMPREHENSIVE HIGH-COMPUTATION CLINICAL DATA SCIENCE SUITE")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"Target Output Directory: {output_dir}\n")

    t_suite_start = time.perf_counter()

    # Hardware & Environment Profiling
    env_info = get_environment_info()

    # Load dataset
    df, X_scaled, y, feature_cols, numeric_cols = prepare_clinical_data()

    # Execute all 6 heavy computational workloads in sequence
    results = {}
    results["manifold"] = benchmark_manifold_learning(X_scaled, y)
    results["tree_ensemble"] = benchmark_tree_ensemble(X_scaled, y, feature_cols)
    results["poly_lasso"] = benchmark_polynomial_lasso(X_scaled, df, numeric_cols)
    results["bootstrap"] = benchmark_massive_bootstrap(df, y)
    results["gmm"] = benchmark_gmm_clustering(X_scaled)
    results["autoencoder"] = benchmark_deep_autoencoder(X_scaled)

    total_runtime = time.perf_counter() - t_suite_start

    # Generate Visualization
    print("\nGenerating 6-panel clinical visualization chart...")
    plot_path = generate_summary_figure(results, df, y, output_dir)

    # Format Text Report
    text_report = format_text_report(env_info, results, total_runtime)

    # Write text report to file
    txt_path = os.path.join(output_dir, "heavy_compute_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    # Also write benchmark_report.txt so standard viewers find it
    bench_txt_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(bench_txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    # Write JSON report
    json_path = os.path.join(output_dir, "heavy_compute_report.json")
    json_export = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_runtime_sec": round(total_runtime, 2),
        "environment": env_info,
        "manifold_learning": {k: v for k, v in results["manifold"].items() if k not in ("X_tsne", "X_kpca")},
        "tree_ensemble": {k: v for k, v in results["tree_ensemble"].items() if k != "all_importances"},
        "polynomial_lasso": {k: v for k, v in results["poly_lasso"].items() if k != "coefs"},
        "bootstrap_resampling": {k: v for k, v in results["bootstrap"].items() if k != "distribution_sample"},
        "gaussian_mixture": results["gmm"],
        "deep_autoencoder": results["autoencoder"]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    # Print the full report to stdout so it appears in logs
    print("\n" + text_report + "\n")

    print("-" * 80)
    print(f"Artifacts successfully written to '{output_dir}':")
    print(f"  - Plain Text Report : {txt_path}")
    print(f"  - Standard Report   : {bench_txt_path}")
    print(f"  - Structured JSON   : {json_path}")
    print(f"  - Visual Chart      : {plot_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
