"""
Regime Clustering Analysis for NQ Signal Feature Database
==========================================================
Discovers market regime clusters and analyzes per-regime signal performance.

Methods: K-Means, GMM, HMM, PCA visualization
Walk-forward validation to test regime-based filtering vs baseline.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── paths ──
ROOT = Path(r"C:\projects\lanto quant\nq quant")
DATA_PATH = ROOT / "data" / "signal_feature_database.parquet"
OUT_DIR = ROOT / "experiments" / "regime_clustering_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── regime features ──
REGIME_FEATURES = [
    "atr_14", "atr_percentile", "fluency_score", "bar_body_ratio",
    "bar_range_atr", "signal_quality", "bias_confidence", "stop_distance_atr",
]


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def perf_metrics(df: pd.DataFrame, label: str = "") -> Dict:
    """Compute R, PPDD, PF for a set of signals."""
    n = len(df)
    if n == 0:
        return dict(label=label, n=0, win_rate=0, avgR=0, totalR=0,
                    maxDD_R=0, PPDD=0, PF=0)
    wins = df[df.outcome_r > 0]
    losses = df[df.outcome_r <= 0]
    win_rate = len(wins) / n * 100
    avgR = df.outcome_r.mean()
    totalR = df.outcome_r.sum()
    # max drawdown in R
    cum = df.outcome_r.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    maxDD_R = abs(dd.min()) if len(dd) > 0 else 0
    PPDD = totalR / maxDD_R if maxDD_R > 0 else (totalR if totalR > 0 else 0)
    gross_profit = wins.outcome_r.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.outcome_r.sum()) if len(losses) > 0 else 0
    PF = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0)
    return dict(label=label, n=n, win_rate=round(win_rate, 1),
                avgR=round(avgR, 4), totalR=round(totalR, 1),
                maxDD_R=round(maxDD_R, 1), PPDD=round(PPDD, 2),
                PF=round(PF, 2))


def print_perf_table(rows: List[Dict], title: str = ""):
    """Pretty-print a performance table."""
    if title:
        log.info(f"\n{'='*80}")
        log.info(f"  {title}")
        log.info(f"{'='*80}")
    df = pd.DataFrame(rows)
    col_order = ["label", "n", "win_rate", "avgR", "totalR", "maxDD_R", "PPDD", "PF"]
    df = df[[c for c in col_order if c in df.columns]]
    log.info(df.to_string(index=False))


def print_section(title: str):
    log.info(f"\n{'#'*80}")
    log.info(f"#  {title}")
    log.info(f"{'#'*80}")


# ═══════════════════════════════════════════════════════════════════
#  1. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare() -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Load data, select regime features, standardize."""
    print_section("1. DATA PREPARATION")
    df = pd.read_parquet(DATA_PATH)
    log.info(f"Loaded {len(df)} signals, {df.shape[1]} columns")
    log.info(f"Date range: {df.bar_time_utc.min()} → {df.bar_time_utc.max()}")

    # add year column
    df["year"] = df.bar_time_utc.dt.year

    # handle NaN in regime features
    for col in REGIME_FEATURES:
        n_null = df[col].isna().sum()
        if n_null > 0:
            log.info(f"  {col}: {n_null} NaN → filling with median")
            df[col] = df[col].fillna(df[col].median())

    X = df[REGIME_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info(f"\nRegime features ({len(REGIME_FEATURES)}):")
    for i, f in enumerate(REGIME_FEATURES):
        log.info(f"  {f:>22s}: mean={X[:, i].mean():.4f}  std={X[:, i].std():.4f}")

    return df, X_scaled, scaler


# ═══════════════════════════════════════════════════════════════════
#  2. K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def run_kmeans(df: pd.DataFrame, X_scaled: np.ndarray) -> int:
    """Run K-Means for k=2..5, pick optimal k."""
    print_section("2. K-MEANS CLUSTERING")

    results = []
    best_k, best_sil = 2, -1
    models = {}

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        inertia = km.inertia_
        sizes = pd.Series(labels).value_counts().sort_index()

        log.info(f"\n  k={k}: silhouette={sil:.4f}  inertia={inertia:.0f}")
        log.info(f"    cluster sizes: {dict(sizes)}")

        results.append(dict(k=k, silhouette=sil, inertia=inertia))
        models[k] = (km, labels)

        if sil > best_sil:
            best_sil = sil
            best_k = k

    log.info(f"\n  >>> Optimal k by silhouette: k={best_k} (sil={best_sil:.4f})")

    # assign best labels
    km_best, labels_best = models[best_k]
    df["km_cluster"] = labels_best

    return best_k


# ═══════════════════════════════════════════════════════════════════
#  3. CLUSTER PROFILING
# ═══════════════════════════════════════════════════════════════════

def profile_clusters(df: pd.DataFrame, cluster_col: str = "km_cluster",
                     method_name: str = "K-Means") -> Dict[int, str]:
    """Profile each cluster: feature means, distributions, naming."""
    print_section(f"3. CLUSTER PROFILING ({method_name})")

    cluster_ids = sorted(df[cluster_col].unique())
    n_clusters = len(cluster_ids)

    # ── feature means per cluster ──
    log.info(f"\n  Feature means by cluster:")
    profile_df = df.groupby(cluster_col)[REGIME_FEATURES].mean()
    log.info(profile_df.round(4).to_string())

    # ── z-score profile (how far each cluster deviates from global mean) ──
    global_means = df[REGIME_FEATURES].mean()
    global_stds = df[REGIME_FEATURES].std()
    z_profiles = (profile_df - global_means) / global_stds
    log.info(f"\n  Z-score profiles (deviation from global mean):")
    log.info(z_profiles.round(2).to_string())

    # ── name clusters ──
    cluster_names = {}
    for cid in cluster_ids:
        z = z_profiles.loc[cid]
        traits = []
        # volatility
        if z["atr_14"] > 0.5:
            traits.append("HighVol")
        elif z["atr_14"] < -0.5:
            traits.append("LowVol")
        else:
            traits.append("MidVol")
        # fluency
        if z["fluency_score"] > 0.3:
            traits.append("Trending")
        elif z["fluency_score"] < -0.3:
            traits.append("Choppy")
        # quality
        if z["signal_quality"] > 0.3:
            traits.append("HiQ")
        elif z["signal_quality"] < -0.3:
            traits.append("LoQ")
        # stop distance
        if z["stop_distance_atr"] > 0.5:
            traits.append("WideStop")
        elif z["stop_distance_atr"] < -0.5:
            traits.append("TightStop")

        name = "/".join(traits) if traits else f"Cluster_{cid}"
        cluster_names[cid] = name
        log.info(f"  Cluster {cid} → \"{name}\" (n={len(df[df[cluster_col]==cid])})")

    # ── distribution of signal_type, grade, session ──
    for cat_col in ["signal_type", "grade", "session"]:
        log.info(f"\n  {cat_col} distribution by cluster:")
        ct = pd.crosstab(df[cluster_col], df[cat_col], normalize="index").round(3)
        log.info(ct.to_string())

    return cluster_names


# ═══════════════════════════════════════════════════════════════════
#  4. PER-CLUSTER PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

def cluster_performance(df: pd.DataFrame, cluster_col: str = "km_cluster",
                        cluster_names: Dict[int, str] = None,
                        method_name: str = "K-Means"):
    """Performance metrics per cluster."""
    print_section(f"4. PER-CLUSTER PERFORMANCE ({method_name})")

    cluster_ids = sorted(df[cluster_col].unique())
    names = cluster_names or {c: f"C{c}" for c in cluster_ids}

    # ── all signals ──
    rows_all = []
    for cid in cluster_ids:
        subset = df[df[cluster_col] == cid].sort_values("bar_time_utc")
        m = perf_metrics(subset, label=f"C{cid}: {names.get(cid, '')}")
        rows_all.append(m)
    rows_all.append(perf_metrics(df.sort_values("bar_time_utc"), label="ALL"))
    print_perf_table(rows_all, f"{method_name} — ALL signals")

    # ── passes_all_filters only ──
    filtered = df[df.passes_all_filters == True]
    if len(filtered) > 0:
        rows_filt = []
        for cid in cluster_ids:
            subset = filtered[filtered[cluster_col] == cid].sort_values("bar_time_utc")
            m = perf_metrics(subset, label=f"C{cid}: {names.get(cid, '')}")
            rows_filt.append(m)
        rows_filt.append(perf_metrics(filtered.sort_values("bar_time_utc"),
                                      label="ALL_FILTERED"))
        print_perf_table(rows_filt, f"{method_name} — passes_all_filters=True")

    return rows_all


# ═══════════════════════════════════════════════════════════════════
#  5. GMM CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def run_gmm(df: pd.DataFrame, X_scaled: np.ndarray):
    """Gaussian Mixture Models for k=2,3,4."""
    print_section("5. GAUSSIAN MIXTURE MODELS")

    best_k, best_bic = 2, np.inf
    models = {}

    for k in [2, 3, 4]:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=5, random_state=42, max_iter=300)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        probs = gmm.predict_proba(X_scaled)
        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))

        sizes = pd.Series(labels).value_counts().sort_index()
        log.info(f"\n  k={k}: BIC={bic:.0f}  AIC={aic:.0f}  silhouette={sil:.4f}")
        log.info(f"    cluster sizes: {dict(sizes)}")

        models[k] = (gmm, labels, probs)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    log.info(f"\n  >>> Optimal k by BIC: k={best_k} (BIC={best_bic:.0f})")

    # assign best labels
    gmm_best, labels_best, probs_best = models[best_k]
    df["gmm_cluster"] = labels_best
    df["gmm_max_prob"] = probs_best.max(axis=1)

    # ── boundary analysis ──
    log.info(f"\n  GMM soft assignment analysis (k={best_k}):")
    log.info(f"    Mean max-probability: {df.gmm_max_prob.mean():.4f}")
    log.info(f"    Signals with max_prob < 0.6 (boundary): {(df.gmm_max_prob < 0.6).sum()}")
    log.info(f"    Signals with max_prob < 0.7 (uncertain): {(df.gmm_max_prob < 0.7).sum()}")

    boundary = df[df.gmm_max_prob < 0.6]
    core = df[df.gmm_max_prob >= 0.8]
    if len(boundary) > 0 and len(core) > 0:
        log.info(f"\n  Boundary signals (max_prob < 0.6) performance:")
        bm = perf_metrics(boundary.sort_values("bar_time_utc"), "Boundary")
        cm = perf_metrics(core.sort_values("bar_time_utc"), "Core (>0.8)")
        print_perf_table([bm, cm], "GMM: Boundary vs Core signals")

    # profile and performance
    gmm_names = profile_clusters(df, "gmm_cluster", "GMM")
    cluster_performance(df, "gmm_cluster", gmm_names, "GMM")

    return best_k, gmm_names


# ═══════════════════════════════════════════════════════════════════
#  6. TEMPORAL STABILITY
# ═══════════════════════════════════════════════════════════════════

def temporal_stability(df: pd.DataFrame, cluster_col: str = "km_cluster",
                       method_name: str = "K-Means"):
    """Check if clusters are stable across years."""
    print_section(f"6. TEMPORAL STABILITY ({method_name})")

    ct = pd.crosstab(df.year, df[cluster_col], normalize="index").round(3)
    log.info(f"\n  Cluster proportions by year:")
    log.info(ct.to_string())

    # coefficient of variation for each cluster's year-over-year proportion
    log.info(f"\n  Stability metrics (CV of cluster proportion across years):")
    for cid in sorted(df[cluster_col].unique()):
        proportions = ct[cid] if cid in ct.columns else pd.Series([0])
        cv = proportions.std() / proportions.mean() if proportions.mean() > 0 else float("inf")
        log.info(f"    Cluster {cid}: mean_prop={proportions.mean():.3f}  "
                 f"std={proportions.std():.3f}  CV={cv:.3f}")

    # per-cluster per-year performance
    log.info(f"\n  Per-cluster per-year avgR:")
    yearly_perf = df.groupby([cluster_col, "year"]).outcome_r.mean().unstack(fill_value=0)
    log.info(yearly_perf.round(3).to_string())


# ═══════════════════════════════════════════════════════════════════
#  7. WALK-FORWARD REGIME FILTER TEST
# ═══════════════════════════════════════════════════════════════════

def walk_forward_test(df: pd.DataFrame, X_scaled: np.ndarray,
                      scaler: StandardScaler):
    """Train on first ~7 years, test on last ~3. Skip toxic clusters."""
    print_section("7. WALK-FORWARD REGIME FILTER TEST")

    years = sorted(df.year.unique())
    n_years = len(years)
    # split: roughly 70/30
    split_year = years[int(n_years * 0.7)]
    train_mask = df.year < split_year
    test_mask = df.year >= split_year

    log.info(f"  Train years: {sorted(df[train_mask].year.unique())}")
    log.info(f"  Test years:  {sorted(df[test_mask].year.unique())}")
    log.info(f"  Train size:  {train_mask.sum()}")
    log.info(f"  Test size:   {test_mask.sum()}")

    X_train = X_scaled[train_mask.values]
    X_test = X_scaled[test_mask.values]
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    for k in [2, 3, 4]:
        log.info(f"\n  {'─'*60}")
        log.info(f"  Walk-forward K-Means k={k}")
        log.info(f"  {'─'*60}")

        # fit on train
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(X_train)

        # predict on train and test
        train_labels = km.predict(X_train)
        test_labels = km.predict(X_test)
        df_train["wf_cluster"] = train_labels
        df_test["wf_cluster"] = test_labels

        # identify toxic clusters from train
        log.info(f"\n  Train-set cluster performance:")
        train_rows = []
        toxic_clusters = []
        for cid in range(k):
            subset = df_train[df_train.wf_cluster == cid].sort_values("bar_time_utc")
            m = perf_metrics(subset, f"Train_C{cid}")
            train_rows.append(m)
            if m["avgR"] < -0.05:
                toxic_clusters.append(cid)
        print_perf_table(train_rows)

        log.info(f"\n  Toxic clusters (avgR < -0.05 in train): {toxic_clusters}")

        # test-set: baseline vs filtered
        test_base = df_test.sort_values("bar_time_utc")
        test_filtered = df_test[~df_test.wf_cluster.isin(toxic_clusters)].sort_values("bar_time_utc")

        base_m = perf_metrics(test_base, "Test_baseline")
        filt_m = perf_metrics(test_filtered, "Test_regime_filtered")

        log.info(f"\n  Test-set comparison:")
        print_perf_table([base_m, filt_m])

        # also test with passes_all_filters
        test_paf = df_test[df_test.passes_all_filters == True]
        if len(test_paf) > 0:
            paf_base = test_paf.sort_values("bar_time_utc")
            paf_filt = test_paf[~test_paf.wf_cluster.isin(toxic_clusters)].sort_values("bar_time_utc")
            base_paf_m = perf_metrics(paf_base, "PAF_baseline")
            filt_paf_m = perf_metrics(paf_filt, "PAF_regime_filtered")
            log.info(f"\n  passes_all_filters=True test-set:")
            print_perf_table([base_paf_m, filt_paf_m])

    # ── compare to simple min_stop_atr_mult filter ──
    log.info(f"\n  {'─'*60}")
    log.info(f"  Comparison: simple stop_distance_atr >= 1.7 filter")
    log.info(f"  {'─'*60}")
    simple_filter = df_test[df_test.stop_distance_atr >= 1.7].sort_values("bar_time_utc")
    simple_m = perf_metrics(simple_filter, "Simple_stop_atr>=1.7")
    all_test_m = perf_metrics(df_test.sort_values("bar_time_utc"), "No_filter")
    print_perf_table([all_test_m, simple_m])


# ═══════════════════════════════════════════════════════════════════
#  8. HIDDEN MARKOV MODEL
# ═══════════════════════════════════════════════════════════════════

def run_hmm(df: pd.DataFrame):
    """HMM on atr_percentile + fluency_score time series."""
    print_section("8. HIDDEN MARKOV MODEL")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.info("  hmmlearn not available, skipping HMM analysis.")
        return

    # sort by time for temporal sequence
    df_sorted = df.sort_values("bar_time_utc").copy()

    # features for HMM
    hmm_features = ["atr_percentile", "fluency_score"]
    X_hmm = df_sorted[hmm_features].values

    # standardize
    scaler_hmm = StandardScaler()
    X_hmm_scaled = scaler_hmm.fit_transform(X_hmm)

    for n_states in [2, 3]:
        log.info(f"\n  HMM with {n_states} states:")
        hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                          n_iter=200, random_state=42)
        hmm.fit(X_hmm_scaled)
        states = hmm.predict(X_hmm_scaled)
        df_sorted[f"hmm_{n_states}"] = states

        log.info(f"    Log-likelihood: {hmm.score(X_hmm_scaled):.1f}")
        log.info(f"    Transition matrix:")
        for i in range(n_states):
            row = "    " + "  ".join(f"{p:.3f}" for p in hmm.transmat_[i])
            log.info(f"      State {i}: [{row}]")

        # state profiles
        log.info(f"\n    State profiles (original scale):")
        for s in range(n_states):
            mask = states == s
            sub = df_sorted[mask]
            log.info(f"      State {s} (n={mask.sum()}):")
            log.info(f"        atr_percentile: {sub.atr_percentile.mean():.3f}")
            log.info(f"        fluency_score:  {sub.fluency_score.mean():.3f}")
            log.info(f"        atr_14:         {sub.atr_14.mean():.2f}")

        # per-state performance
        rows = []
        for s in range(n_states):
            sub = df_sorted[df_sorted[f"hmm_{n_states}"] == s].sort_values("bar_time_utc")
            m = perf_metrics(sub, f"HMM_State_{s}")
            rows.append(m)
        rows.append(perf_metrics(df_sorted, "ALL"))
        print_perf_table(rows, f"HMM {n_states}-state performance")

        # state persistence (avg run length)
        runs = []
        current_state = states[0]
        run_len = 1
        for s in states[1:]:
            if s == current_state:
                run_len += 1
            else:
                runs.append((current_state, run_len))
                current_state = s
                run_len = 1
        runs.append((current_state, run_len))

        for s in range(n_states):
            s_runs = [r[1] for r in runs if r[0] == s]
            if s_runs:
                log.info(f"    State {s} avg run length: {np.mean(s_runs):.1f} signals "
                         f"(median={np.median(s_runs):.0f}, max={max(s_runs)})")


# ═══════════════════════════════════════════════════════════════════
#  9. PCA VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def pca_visualization(df: pd.DataFrame, X_scaled: np.ndarray,
                      cluster_col: str = "km_cluster"):
    """PCA 2D scatter plots."""
    print_section("9. PCA VISUALIZATION")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    log.info(f"  PCA explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, "
             f"total={sum(evr):.3f}")

    # loadings
    log.info(f"\n  PCA loadings:")
    for i, feat in enumerate(REGIME_FEATURES):
        log.info(f"    {feat:>22s}: PC1={pca.components_[0, i]:+.3f}  "
                 f"PC2={pca.components_[1, i]:+.3f}")

    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    # ── plot 1: colored by cluster ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # cluster coloring
    ax = axes[0]
    clusters = df[cluster_col].values
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10",
                         alpha=0.3, s=5)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("PCA colored by K-Means cluster")
    plt.colorbar(scatter, ax=ax)

    # outcome coloring
    ax = axes[1]
    win_mask = df.outcome_r > 0
    ax.scatter(X_pca[~win_mask, 0], X_pca[~win_mask, 1], c="red",
               alpha=0.15, s=5, label="Loss")
    ax.scatter(X_pca[win_mask, 0], X_pca[win_mask, 1], c="green",
               alpha=0.15, s=5, label="Win")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("PCA colored by outcome")
    ax.legend()

    # avgR heatmap
    ax = axes[2]
    scatter2 = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=df.outcome_r.clip(-1, 3), cmap="RdYlGn",
                          alpha=0.3, s=5, vmin=-1, vmax=3)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("PCA colored by outcome_r")
    plt.colorbar(scatter2, ax=ax)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {OUT_DIR / 'pca_scatter.png'}")

    # ── plot 2: cluster feature heatmap ──
    fig, ax = plt.subplots(figsize=(12, 5))
    profile_df = df.groupby(cluster_col)[REGIME_FEATURES].mean()
    global_means = df[REGIME_FEATURES].mean()
    global_stds = df[REGIME_FEATURES].std()
    z_profiles = (profile_df - global_means) / global_stds
    sns.heatmap(z_profiles, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, xticklabels=[f.replace("_", "\n") for f in REGIME_FEATURES])
    ax.set_title("Cluster Z-score profiles")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {OUT_DIR / 'cluster_heatmap.png'}")

    # ── plot 3: per-cluster performance bars ──
    cluster_ids = sorted(df[cluster_col].unique())
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics_names = ["win_rate", "avgR", "totalR", "PPDD"]
    for ax, metric in zip(axes, metrics_names):
        vals = []
        for cid in cluster_ids:
            sub = df[df[cluster_col] == cid].sort_values("bar_time_utc")
            m = perf_metrics(sub)
            vals.append(m[metric])
        colors = ["green" if v > 0 else "red" for v in vals]
        ax.bar([f"C{c}" for c in cluster_ids], vals, color=colors, alpha=0.7)
        ax.set_title(metric)
        ax.axhline(0, color="black", linewidth=0.5)
    plt.suptitle("Per-cluster performance metrics")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cluster_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {OUT_DIR / 'cluster_performance.png'}")


# ═══════════════════════════════════════════════════════════════════
#  10. ADDITIONAL DEEP DIVE
# ═══════════════════════════════════════════════════════════════════

def feature_correlation_analysis(df: pd.DataFrame):
    """How do regime features correlate with outcome?"""
    print_section("10. FEATURE-OUTCOME CORRELATIONS")

    log.info(f"\n  Pearson correlation of regime features with outcome_r:")
    for feat in REGIME_FEATURES:
        corr = df[feat].corr(df.outcome_r)
        log.info(f"    {feat:>22s}: r={corr:+.4f}")

    # for filtered signals
    filtered = df[df.passes_all_filters == True]
    if len(filtered) > 50:
        log.info(f"\n  Same for passes_all_filters=True (n={len(filtered)}):")
        for feat in REGIME_FEATURES:
            corr = filtered[feat].corr(filtered.outcome_r)
            log.info(f"    {feat:>22s}: r={corr:+.4f}")


def interaction_analysis(df: pd.DataFrame):
    """Check if combining regime cluster with existing filters helps."""
    print_section("11. INTERACTION: REGIME × EXISTING FILTERS")

    if "km_cluster" not in df.columns:
        return

    filtered = df[df.passes_all_filters == True]
    if len(filtered) == 0:
        log.info("  No filtered signals to analyze.")
        return

    cluster_ids = sorted(df.km_cluster.unique())
    rows = []
    for cid in cluster_ids:
        sub = filtered[filtered.km_cluster == cid].sort_values("bar_time_utc")
        m = perf_metrics(sub, f"PAF+C{cid}")
        rows.append(m)
    rows.append(perf_metrics(filtered.sort_values("bar_time_utc"), "PAF_all"))
    print_perf_table(rows, "passes_all_filters=True × K-Means cluster")

    # best cluster combo
    log.info(f"\n  Which clusters to keep when combined with existing filters?")
    for cid in cluster_ids:
        sub = filtered[filtered.km_cluster == cid].sort_values("bar_time_utc")
        m = perf_metrics(sub)
        verdict = "KEEP" if m["avgR"] > 0 else "DROP"
        log.info(f"    Cluster {cid}: avgR={m['avgR']:.4f} → {verdict}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 80)
    log.info("  NQ REGIME CLUSTERING ANALYSIS")
    log.info("=" * 80)

    # 1. Load & prepare
    df, X_scaled, scaler = load_and_prepare()

    # 2. K-Means
    best_k = run_kmeans(df, X_scaled)

    # 3. Cluster profiling
    km_names = profile_clusters(df, "km_cluster", "K-Means")

    # 4. Per-cluster performance
    cluster_performance(df, "km_cluster", km_names, "K-Means")

    # 5. GMM
    gmm_k, gmm_names = run_gmm(df, X_scaled)

    # 6. Temporal stability
    temporal_stability(df, "km_cluster", "K-Means")

    # 7. Walk-forward
    walk_forward_test(df, X_scaled, scaler)

    # 8. HMM
    run_hmm(df)

    # 9. PCA viz
    pca_visualization(df, X_scaled, "km_cluster")

    # 10. Correlations
    feature_correlation_analysis(df)

    # 11. Interaction
    interaction_analysis(df)

    # ── summary ──
    print_section("EXECUTIVE SUMMARY")
    log.info("""
  This analysis investigated whether distinct market REGIMES exist in the
  NQ signal database, and whether regime-based filtering can improve
  trading performance beyond the current simple stop_distance_atr filter.

  Key outputs saved to: %s
    - pca_scatter.png      : PCA 2D projections (cluster, outcome, R)
    - cluster_heatmap.png  : Z-score feature profiles per cluster
    - cluster_performance.png : Performance bar charts
    """, OUT_DIR)

    log.info("=" * 80)
    log.info("  ANALYSIS COMPLETE")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
