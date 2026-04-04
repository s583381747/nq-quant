"""
Sequential / Contextual Features Analysis
==========================================
Does recent trading history affect future signal quality?
- Autocorrelation in outcomes
- Streak effects (hot hand / gambler's fallacy)
- Daily position effects (1st vs 2nd vs 3rd signal)
- Time gap effects
- 0-for-2 rule validation
- XGBoost walk-forward with sequential features

Every table includes: R, PPDD (R/MaxDD), PF.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_feature_database.parquet"

# ═══════════════════════════ HELPERS ═══════════════════════════

def compute_metrics(outcomes: np.ndarray) -> dict:
    """Compute standard trading metrics from outcome_r array."""
    n = len(outcomes)
    if n == 0:
        return dict(count=0, avg_r=np.nan, med_r=np.nan, wr=np.nan,
                    pf=np.nan, total_r=np.nan, ppdd=np.nan, max_dd=np.nan)

    total_r = float(np.sum(outcomes))
    avg_r = float(np.mean(outcomes))
    med_r = float(np.median(outcomes))
    wr = float(np.mean(outcomes > 0)) * 100

    wins = outcomes[outcomes > 0]
    losses = outcomes[outcomes < 0]
    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = sum_wins / sum_losses if sum_losses > 0 else (np.inf if sum_wins > 0 else 0.0)

    cum = np.cumsum(outcomes)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ppdd = total_r / max_dd if max_dd > 0 else (np.inf if total_r > 0 else 0.0)

    return dict(count=n, avg_r=round(avg_r, 4), med_r=round(med_r, 4),
                wr=round(wr, 1), pf=round(pf, 3), total_r=round(total_r, 2),
                ppdd=round(ppdd, 3), max_dd=round(max_dd, 2))


def metrics_row(label: str, outcomes: np.ndarray) -> dict:
    m = compute_metrics(outcomes)
    m["bin"] = label
    return m


def metrics_table(groups: dict) -> pd.DataFrame:
    rows = [metrics_row(label, grp["outcome_r"].values) for label, grp in groups.items()]
    df = pd.DataFrame(rows)
    cols = ["bin", "count", "avg_r", "med_r", "wr", "pf", "total_r", "ppdd", "max_dd"]
    return df[[c for c in cols if c in df.columns]]


def runs_test(x: np.ndarray) -> tuple:
    """Wald-Wolfowitz runs test for randomness of binary sequence.
    Returns (z_stat, p_value)."""
    n = len(x)
    if n < 20:
        return (np.nan, np.nan)
    n1 = np.sum(x == 1)
    n0 = np.sum(x == 0)
    if n1 == 0 or n0 == 0:
        return (np.nan, np.nan)

    # Count runs
    runs = 1
    for i in range(1, n):
        if x[i] != x[i - 1]:
            runs += 1

    # Expected runs and variance
    e_runs = 1 + (2 * n1 * n0) / (n1 + n0)
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))

    if var_runs <= 0:
        return (np.nan, np.nan)

    z = (runs - e_runs) / np.sqrt(var_runs)
    p = 2 * stats.norm.sf(abs(z))
    return (z, p)


# ═══════════════════════════ BUILD SEQUENTIAL FEATURES ═══════════════════════════

def build_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all sequential/contextual features. Only uses PAST data."""
    df = df.sort_values("bar_time_et").reset_index(drop=True)
    n = len(df)

    # Extract date and direction
    df["date"] = df["bar_time_et"].dt.date

    # Determine session for each signal (using hour_et)
    # Sessions: asia (18-03), london (03-09:30), ny (09:30-16)
    # Already have 'session' column, use it

    # ─── Previous signal outcomes ───
    df["prev_1_outcome_r"] = df["outcome_r"].shift(1)
    df["prev_2_outcome_r"] = df["outcome_r"].shift(2)
    df["prev_1_win"] = (df["outcome_r"].shift(1) > 0).astype(float)
    df["prev_1_loss"] = (df["outcome_r"].shift(1) < 0).astype(float)

    # ─── Streaks (wins/losses ending at previous signal) ───
    streak_wins = np.zeros(n, dtype=float)
    streak_losses = np.zeros(n, dtype=float)
    outcomes = df["outcome_r"].values
    for i in range(1, n):
        # Look backward from i-1
        w_count = 0
        l_count = 0
        j = i - 1
        while j >= 0 and outcomes[j] > 0:
            w_count += 1
            j -= 1
        j = i - 1
        while j >= 0 and outcomes[j] <= 0:
            l_count += 1
            j -= 1
        # Streak wins: consecutive wins ending at i-1
        if outcomes[i - 1] > 0:
            streak_wins[i] = w_count
            streak_losses[i] = 0
        else:
            streak_wins[i] = 0
            streak_losses[i] = l_count
    df["streak_wins"] = streak_wins
    df["streak_losses"] = streak_losses

    # ─── Daily features ───
    daily_signal_number = np.zeros(n, dtype=int)
    daily_cumulative_r = np.zeros(n, dtype=float)
    dates = df["date"].values
    for i in range(n):
        d = dates[i]
        # Count how many signals with same date came before
        count = 0
        cum_r = 0.0
        j = i - 1
        while j >= 0 and dates[j] == d:
            count += 1
            cum_r += outcomes[j]
            j -= 1
        daily_signal_number[i] = count + 1  # 1-indexed
        daily_cumulative_r[i] = cum_r
    df["daily_signal_number"] = daily_signal_number
    df["daily_cumulative_r"] = daily_cumulative_r

    # ─── Time since last signal ───
    # bar_time_et is datetime64[us] (microseconds), convert to minutes
    times = df["bar_time_et"].values.astype("int64") / 1e6 / 60  # microseconds → minutes
    time_since_last = np.full(n, np.nan)
    for i in range(1, n):
        time_since_last[i] = times[i] - times[i - 1]
    df["time_since_last_signal"] = time_since_last

    # ─── Time since last win ───
    time_since_last_win = np.full(n, np.nan)
    for i in range(1, n):
        j = i - 1
        while j >= 0:
            if outcomes[j] > 0:
                time_since_last_win[i] = times[i] - times[j]
                break
            j -= 1
    df["time_since_last_win"] = time_since_last_win

    # ─── Same direction streak ───
    directions = df["signal_dir"].values
    same_dir_streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        count = 0
        j = i - 1
        while j >= 0 and directions[j] == directions[i]:
            count += 1
            j -= 1
        same_dir_streak[i] = count
    df["same_direction_streak"] = same_dir_streak

    # ─── Session signal number ───
    sessions = df["session"].values
    session_signal_number = np.zeros(n, dtype=int)
    for i in range(n):
        d = dates[i]
        s = sessions[i]
        count = 0
        j = i - 1
        while j >= 0 and dates[j] == d and sessions[j] == s:
            count += 1
            j -= 1
        session_signal_number[i] = count + 1
    df["session_signal_number"] = session_signal_number

    # ─── Rolling win rate (last 5, last 10) ───
    rolling_wr_5 = np.full(n, np.nan)
    rolling_wr_10 = np.full(n, np.nan)
    rolling_avg_r_5 = np.full(n, np.nan)
    for i in range(5, n):
        rolling_wr_5[i] = np.mean(outcomes[i-5:i] > 0)
        rolling_avg_r_5[i] = np.mean(outcomes[i-5:i])
    for i in range(10, n):
        rolling_wr_10[i] = np.mean(outcomes[i-10:i] > 0)
    df["rolling_wr_5"] = rolling_wr_5
    df["rolling_wr_10"] = rolling_wr_10
    df["rolling_avg_r_5"] = rolling_avg_r_5

    return df


# ═══════════════════════════ ANALYSIS FUNCTIONS ═══════════════════════════

def section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def correlation_analysis(df: pd.DataFrame, seq_features: list):
    """Correlation of sequential features with outcomes."""
    section("1. CORRELATION ANALYSIS")

    print(f"{'Feature':<30} {'corr(R)':<10} {'p(R)':<12} {'corr(label)':<12} {'p(label)':<12}")
    print("-" * 76)

    results = []
    for feat in seq_features:
        valid = df[[feat, "outcome_r", "outcome_label"]].dropna()
        if len(valid) < 30:
            continue
        r_corr, r_p = spearmanr(valid[feat], valid["outcome_r"])
        l_corr, l_p = spearmanr(valid[feat], valid["outcome_label"])
        results.append((feat, r_corr, r_p, l_corr, l_p))
        sig_r = "*" if r_p < 0.05 else " "
        sig_l = "*" if l_p < 0.05 else " "
        print(f"{feat:<30} {r_corr:>+.4f} {sig_r} {r_p:>.2e}   {l_corr:>+.4f} {sig_l} {l_p:>.2e}")

    # Sort by abs correlation with outcome_r
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 5 by |corr(outcome_r)|:")
    for feat, r, rp, l, lp in results[:5]:
        print(f"    {feat:<30} corr={r:+.4f}  p={rp:.2e}")


def conditional_performance(df: pd.DataFrame, label: str = "ALL"):
    """Conditional performance breakdowns."""
    section(f"2. CONDITIONAL PERFORMANCE ({label})")

    # ─── After win vs after loss ───
    print("  A. Performance AFTER previous signal outcome:")
    after_win = df[df["prev_1_win"] == 1]
    after_loss = df[df["prev_1_loss"] == 1]
    after_neither = df[(df["prev_1_win"] != 1) & (df["prev_1_loss"] != 1)]
    tbl = metrics_table({
        "after_win": after_win,
        "after_loss": after_loss,
        "after_neutral/NaN": after_neither,
        "ALL": df
    })
    print(tbl.to_string(index=False))

    # Stat test
    if len(after_win) > 10 and len(after_loss) > 10:
        t, p = stats.mannwhitneyu(after_win["outcome_r"].dropna(),
                                   after_loss["outcome_r"].dropna(),
                                   alternative="two-sided")
        print(f"  Mann-Whitney after_win vs after_loss: U={t:.0f}, p={p:.4f}")

    # ─── By daily signal number ───
    print("\n  B. Performance by DAILY SIGNAL NUMBER:")
    groups = {}
    for num in sorted(df["daily_signal_number"].unique()):
        sub = df[df["daily_signal_number"] == num]
        if len(sub) >= 10:
            lbl = f"signal_{num}" if num <= 5 else "signal_6+"
            if lbl in groups:
                groups[lbl] = pd.concat([groups[lbl], sub])
            else:
                groups[lbl] = sub
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # ─── By streak length ───
    print("\n  C. Performance by STREAK (preceding signals):")
    groups = {
        "after_0_wins": df[df["streak_wins"] == 0],
        "after_1_win": df[df["streak_wins"] == 1],
        "after_2+_wins": df[df["streak_wins"] >= 2],
        "after_1_loss": df[df["streak_losses"] == 1],
        "after_2_losses": df[df["streak_losses"] == 2],
        "after_3+_losses": df[df["streak_losses"] >= 3],
    }
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # ─── By time since last signal ───
    print("\n  D. Performance by TIME SINCE LAST SIGNAL:")
    valid = df.dropna(subset=["time_since_last_signal"])
    groups = {
        "<30min": valid[valid["time_since_last_signal"] < 30],
        "30-60min": valid[(valid["time_since_last_signal"] >= 30) & (valid["time_since_last_signal"] < 60)],
        "1-4h": valid[(valid["time_since_last_signal"] >= 60) & (valid["time_since_last_signal"] < 240)],
        "4h+": valid[valid["time_since_last_signal"] >= 240],
    }
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # ─── By session signal number ───
    print("\n  E. Performance by SESSION SIGNAL NUMBER:")
    groups = {}
    for num in range(1, 8):
        sub = df[df["session_signal_number"] == num]
        if len(sub) >= 10:
            groups[f"session_sig_{num}"] = sub
    if len(df[df["session_signal_number"] >= 8]) >= 10:
        groups["session_sig_8+"] = df[df["session_signal_number"] >= 8]
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))


def hot_hand_analysis(df: pd.DataFrame, label: str = "ALL"):
    """Test for hot hand / gambler's fallacy / autocorrelation."""
    section(f"3. HOT HAND / GAMBLER'S FALLACY TEST ({label})")

    outcomes = df["outcome_r"].values
    binary = (outcomes > 0).astype(int)

    # ─── Runs test ───
    z, p = runs_test(binary)
    print(f"  Wald-Wolfowitz Runs Test:")
    print(f"    Z-statistic: {z:.4f}")
    print(f"    P-value:     {p:.4f}")
    if p < 0.05:
        if z < 0:
            print(f"    SIGNIFICANT: outcomes cluster (hot hand / momentum)")
        else:
            print(f"    SIGNIFICANT: outcomes alternate (mean reversion)")
    else:
        print(f"    Not significant: outcomes appear random")

    # ─── Autocorrelation lag 1 ───
    valid_outcomes = outcomes[~np.isnan(outcomes)]
    if len(valid_outcomes) > 10:
        ac1 = np.corrcoef(valid_outcomes[:-1], valid_outcomes[1:])[0, 1]
        # Significance of autocorrelation
        n_ac = len(valid_outcomes) - 1
        se = 1 / np.sqrt(n_ac)
        z_ac = ac1 / se
        p_ac = 2 * stats.norm.sf(abs(z_ac))
        print(f"\n  Autocorrelation (lag 1) of outcome_r:")
        print(f"    r = {ac1:.4f}, z = {z_ac:.4f}, p = {p_ac:.4f}")
        if p_ac < 0.05:
            print(f"    SIGNIFICANT autocorrelation detected!")
        else:
            print(f"    No significant autocorrelation")

    # ─── After big win (>2R) ───
    print(f"\n  After BIG WIN (outcome_r > 2R):")
    big_win_idx = np.where(outcomes > 2)[0]
    next_after_big = []
    for idx in big_win_idx:
        if idx + 1 < len(outcomes):
            next_after_big.append(outcomes[idx + 1])
    if len(next_after_big) >= 5:
        next_arr = np.array(next_after_big)
        print(f"    N = {len(next_arr)}")
        print(f"    Avg next R: {np.mean(next_arr):.4f}")
        print(f"    Win rate:   {np.mean(next_arr > 0) * 100:.1f}%")
        print(f"    Baseline avg R: {np.mean(outcomes):.4f}")
        t, p = stats.mannwhitneyu(next_arr, outcomes, alternative="two-sided")
        print(f"    vs baseline: U={t:.0f}, p={p:.4f}")
    else:
        print(f"    Not enough data ({len(next_after_big)} signals)")

    # ─── After loss ───
    print(f"\n  After LOSS (outcome_r <= 0):")
    loss_idx = np.where(outcomes <= 0)[0]
    next_after_loss = []
    for idx in loss_idx:
        if idx + 1 < len(outcomes):
            next_after_loss.append(outcomes[idx + 1])
    next_arr = np.array(next_after_loss)
    print(f"    N = {len(next_arr)}")
    print(f"    Avg next R: {np.mean(next_arr):.4f}")
    print(f"    Win rate:   {np.mean(next_arr > 0) * 100:.1f}%")


def daily_pattern_analysis(df: pd.DataFrame, label: str = "ALL"):
    """Daily pattern and 0-for-N rule validation."""
    section(f"4. DAILY PATTERN ANALYSIS ({label})")

    # ─── 1st vs later signals ───
    print("  A. 1st signal of day vs later signals:")
    first = df[df["daily_signal_number"] == 1]
    later = df[df["daily_signal_number"] > 1]
    tbl = metrics_table({"1st_of_day": first, "2nd+_of_day": later, "ALL": df})
    print(tbl.to_string(index=False))
    if len(first) > 10 and len(later) > 10:
        t, p = stats.mannwhitneyu(first["outcome_r"].dropna(),
                                   later["outcome_r"].dropna(),
                                   alternative="two-sided")
        print(f"  Mann-Whitney: U={t:.0f}, p={p:.4f}")

    # ─── Cumulative daily R effect ───
    print("\n  B. Effect of DAILY CUMULATIVE R on next signal:")
    valid = df.dropna(subset=["daily_cumulative_r"])
    valid = valid[valid["daily_signal_number"] > 1]  # Only after 1st signal
    groups = {
        "cum_r < -1": valid[valid["daily_cumulative_r"] < -1],
        "-1 <= cum_r < 0": valid[(valid["daily_cumulative_r"] >= -1) & (valid["daily_cumulative_r"] < 0)],
        "0 <= cum_r < 1": valid[(valid["daily_cumulative_r"] >= 0) & (valid["daily_cumulative_r"] < 1)],
        "cum_r >= 1": valid[valid["daily_cumulative_r"] >= 1],
    }
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # ─── 0-for-N rule validation ───
    print("\n  C. 0-FOR-N RULE VALIDATION:")
    print("     Simulating what happens after N consecutive losses on same day")

    # Group by date, replay sequences
    dates = df["date"].unique()
    results_by_n = {0: [], 1: [], 2: [], 3: [], 4: []}

    for d in dates:
        day_signals = df[df["date"] == d].sort_values("bar_time_et")
        consec_losses = 0
        for _, row in day_signals.iterrows():
            if consec_losses in results_by_n:
                results_by_n[consec_losses].append(row["outcome_r"])
            if row["outcome_r"] <= 0:
                consec_losses += 1
            else:
                consec_losses = 0

    print(f"  {'After N losses':<20} {'Count':<8} {'AvgR':<10} {'WinRate':<10} {'TotalR':<10}")
    print("  " + "-" * 58)
    for n_losses in range(5):
        arr = np.array(results_by_n[n_losses])
        if len(arr) > 0:
            avg_r = np.mean(arr)
            wr = np.mean(arr > 0) * 100
            tot = np.sum(arr)
            print(f"  after_{n_losses}_losses    {len(arr):<8} {avg_r:>+.4f}    {wr:>5.1f}%    {tot:>+.2f}")

    # Calculate R saved/lost by different N-for-N rules
    print(f"\n  D. R IMPACT OF DIFFERENT STOP RULES:")
    print(f"  (How much R do we avoid by stopping after N consecutive losses?)")
    for stop_after in [1, 2, 3, 4]:
        r_avoided = 0
        signals_avoided = 0
        for d in dates:
            day_signals = df[df["date"] == d].sort_values("bar_time_et")
            consec_losses = 0
            for _, row in day_signals.iterrows():
                if consec_losses >= stop_after:
                    r_avoided += row["outcome_r"]
                    signals_avoided += 1
                if row["outcome_r"] <= 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
        print(f"  Stop after {stop_after} losses: {signals_avoided:>5} signals skipped, "
              f"R avoided = {r_avoided:>+.2f} "
              f"({'GOOD - skip bad R' if r_avoided < 0 else 'BAD - skip good R'})")


def xgboost_analysis(df: pd.DataFrame, seq_features: list):
    """Walk-forward XGBoost with and without sequential features."""
    section("5. XGBOOST WALK-FORWARD WITH SEQUENTIAL FEATURES")

    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    except ImportError:
        print("  XGBoost not available, skipping.")
        return

    # Identify original features (non-sequential, non-meta)
    meta_cols = ["bar_time_utc", "bar_time_et", "signal_type", "entry_price",
                 "model_stop", "irl_target", "session", "sub_session", "grade",
                 "hit_tp", "hit_sl", "outcome_r", "max_favorable_excursion",
                 "max_adverse_excursion", "bars_to_outcome", "outcome_label",
                 "date", "signal_dir"]
    blocked_cols = [c for c in df.columns if c.startswith("blocked_by_")]
    exclude = set(meta_cols + blocked_cols + ["passes_all_filters"] + seq_features)

    orig_features = [c for c in df.columns if c not in exclude
                     and df[c].dtype in ["float64", "int64", "bool"]
                     and c not in seq_features]

    # Convert bools to int
    for c in orig_features + seq_features:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(int)

    # Binary label: 1 = win (outcome_r > 0), 0 = loss
    df["y"] = (df["outcome_r"] > 0).astype(int)

    # Walk-forward: train on years [start..year-1], test on year
    df["year"] = df["bar_time_et"].dt.year
    years = sorted(df["year"].unique())

    results_orig = []
    results_seq = []
    importances_orig = {}
    importances_seq = {}

    for test_year in years:
        if test_year < 2018:  # Need enough training data
            continue

        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()

        if len(train) < 100 or len(test) < 50:
            continue

        # ─── Original features only ───
        X_train_orig = train[orig_features].fillna(0)
        X_test_orig = test[orig_features].fillna(0)
        y_train = train["y"].values
        y_test = test["y"].values

        model_orig = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric="logloss"
        )
        model_orig.fit(X_train_orig, y_train)
        pred_orig = model_orig.predict(X_test_orig)
        prob_orig = model_orig.predict_proba(X_test_orig)[:, 1]

        prec_orig = precision_score(y_test, pred_orig, zero_division=0)
        auc_orig = roc_auc_score(y_test, prob_orig) if len(np.unique(y_test)) > 1 else 0.5
        results_orig.append({"year": test_year, "precision": prec_orig, "auc": auc_orig,
                             "n_test": len(test)})

        # Track feature importances
        for feat, imp in zip(orig_features, model_orig.feature_importances_):
            importances_orig[feat] = importances_orig.get(feat, 0) + imp

        # ─── Original + sequential features ───
        all_features = orig_features + seq_features
        X_train_all = train[all_features].fillna(0)
        X_test_all = test[all_features].fillna(0)

        model_seq = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric="logloss"
        )
        model_seq.fit(X_train_all, y_train)
        pred_seq = model_seq.predict(X_test_all)
        prob_seq = model_seq.predict_proba(X_test_all)[:, 1]

        prec_seq = precision_score(y_test, pred_seq, zero_division=0)
        auc_seq = roc_auc_score(y_test, prob_seq) if len(np.unique(y_test)) > 1 else 0.5
        results_seq.append({"year": test_year, "precision": prec_seq, "auc": auc_seq,
                            "n_test": len(test)})

        for feat, imp in zip(all_features, model_seq.feature_importances_):
            importances_seq[feat] = importances_seq.get(feat, 0) + imp

    # Print results
    print("  A. Walk-forward results — ORIGINAL features only:")
    df_orig = pd.DataFrame(results_orig)
    print(df_orig.to_string(index=False))
    print(f"  Average precision: {df_orig['precision'].mean():.4f}")
    print(f"  Average AUC:       {df_orig['auc'].mean():.4f}")

    print(f"\n  B. Walk-forward results — ORIGINAL + SEQUENTIAL features:")
    df_seq = pd.DataFrame(results_seq)
    print(df_seq.to_string(index=False))
    print(f"  Average precision: {df_seq['precision'].mean():.4f}")
    print(f"  Average AUC:       {df_seq['auc'].mean():.4f}")

    # Delta
    prec_delta = df_seq['precision'].mean() - df_orig['precision'].mean()
    auc_delta = df_seq['auc'].mean() - df_orig['auc'].mean()
    print(f"\n  C. DELTA (seq - orig):")
    print(f"    Precision: {prec_delta:+.4f}")
    print(f"    AUC:       {auc_delta:+.4f}")

    # Feature importance ranking
    print(f"\n  D. Top 20 Feature Importances (with sequential features):")
    sorted_imp = sorted(importances_seq.items(), key=lambda x: x[1], reverse=True)
    n_years = len(results_seq)
    print(f"  {'Rank':<6} {'Feature':<35} {'Avg Importance':<15} {'Sequential?'}")
    print("  " + "-" * 70)
    for rank, (feat, imp) in enumerate(sorted_imp[:20], 1):
        is_seq = "YES" if feat in seq_features else ""
        print(f"  {rank:<6} {feat:<35} {imp/n_years:>10.4f}       {is_seq}")

    # Count sequential features in top 20
    top20_feats = [f for f, _ in sorted_imp[:20]]
    seq_in_top20 = [f for f in top20_feats if f in seq_features]
    print(f"\n  Sequential features in top 20: {len(seq_in_top20)}/{len(seq_features)}")
    if seq_in_top20:
        print(f"  Which ones: {', '.join(seq_in_top20)}")


def filtered_analysis(df: pd.DataFrame):
    """Run conditional analysis on passes_all_filters subset."""
    filt = df[df["passes_all_filters"]].copy()
    if len(filt) < 30:
        print("  Not enough filtered signals for meaningful analysis")
        return

    section("6. FILTERED SIGNALS ANALYSIS (passes_all_filters=True)")
    print(f"  N = {len(filt)} filtered signals\n")

    # After win vs after loss
    print("  A. After previous signal outcome (FILTERED):")
    # Need to use sequential features already computed on full dataset
    after_win = filt[filt["prev_1_win"] == 1]
    after_loss = filt[filt["prev_1_loss"] == 1]
    tbl = metrics_table({
        "after_win": after_win,
        "after_loss": after_loss,
        "ALL_filtered": filt,
    })
    print(tbl.to_string(index=False))

    # By daily signal number
    print("\n  B. By daily signal number (FILTERED):")
    groups = {}
    for num in sorted(filt["daily_signal_number"].unique()):
        sub = filt[filt["daily_signal_number"] == num]
        if len(sub) >= 5:
            lbl = f"signal_{num}" if num <= 4 else "signal_5+"
            if lbl in groups:
                groups[lbl] = pd.concat([groups[lbl], sub])
            else:
                groups[lbl] = sub
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # By streak
    print("\n  C. By streak (FILTERED):")
    groups = {
        "after_0_wins": filt[filt["streak_wins"] == 0],
        "after_1+_wins": filt[filt["streak_wins"] >= 1],
        "after_1_loss": filt[filt["streak_losses"] == 1],
        "after_2+_losses": filt[filt["streak_losses"] >= 2],
    }
    groups = {k: v for k, v in groups.items() if len(v) >= 5}
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # Time since last signal
    print("\n  D. By time since last signal (FILTERED):")
    valid = filt.dropna(subset=["time_since_last_signal"])
    groups = {
        "<30min": valid[valid["time_since_last_signal"] < 30],
        "30-60min": valid[(valid["time_since_last_signal"] >= 30) & (valid["time_since_last_signal"] < 60)],
        "1-4h": valid[(valid["time_since_last_signal"] >= 60) & (valid["time_since_last_signal"] < 240)],
        "4h+": valid[valid["time_since_last_signal"] >= 240],
    }
    groups = {k: v for k, v in groups.items() if len(v) >= 5}
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))


def practical_recommendations(df: pd.DataFrame):
    """Synthesize findings into actionable recommendations."""
    section("7. PRACTICAL RECOMMENDATIONS")

    filt = df[df["passes_all_filters"]].copy()

    print("  A. SEQUENTIAL RULE CANDIDATES (on filtered signals):")
    print()

    # Rule 1: Skip if daily_cumulative_r < threshold
    print("  Rule 1: Tilt guard — skip if daily cumulative R < threshold")
    for thresh in [-2.0, -1.5, -1.0, -0.5]:
        sub_keep = filt[filt["daily_cumulative_r"] >= thresh]
        sub_skip = filt[filt["daily_cumulative_r"] < thresh]
        m_keep = compute_metrics(sub_keep["outcome_r"].values)
        m_skip = compute_metrics(sub_skip["outcome_r"].values)
        print(f"    Threshold {thresh:>+.1f}: keep={m_keep['count']:>4} avgR={m_keep['avg_r']:>+.4f} | "
              f"skip={m_skip['count']:>4} avgR={m_skip['avg_r']:>+.4f}")

    # Rule 2: Skip if time_since_last_signal < threshold
    print("\n  Rule 2: Cooldown — skip if time since last signal < threshold")
    valid = filt.dropna(subset=["time_since_last_signal"])
    for thresh in [5, 10, 15, 20, 30]:
        sub_keep = valid[valid["time_since_last_signal"] >= thresh]
        sub_skip = valid[valid["time_since_last_signal"] < thresh]
        m_keep = compute_metrics(sub_keep["outcome_r"].values)
        m_skip = compute_metrics(sub_skip["outcome_r"].values)
        print(f"    >={thresh:>2}min: keep={m_keep['count']:>4} avgR={m_keep['avg_r']:>+.4f} | "
              f"skip={m_skip['count']:>4} avgR={m_skip['avg_r']:>+.4f}")

    # Rule 3: Max daily signals
    print("\n  Rule 3: Max daily signals — skip after Nth signal")
    for max_n in [1, 2, 3, 4, 5]:
        sub_keep = filt[filt["daily_signal_number"] <= max_n]
        sub_skip = filt[filt["daily_signal_number"] > max_n]
        m_keep = compute_metrics(sub_keep["outcome_r"].values)
        m_skip = compute_metrics(sub_skip["outcome_r"].values)
        r_delta = m_keep['total_r'] - compute_metrics(filt["outcome_r"].values)['total_r']
        print(f"    Max {max_n}: keep={m_keep['count']:>4} avgR={m_keep['avg_r']:>+.4f} totalR={m_keep['total_r']:>+.2f} "
              f"PPDD={m_keep['ppdd']:.3f} | skip={m_skip['count']:>4} avgR={m_skip['avg_r']:>+.4f}")

    # Rule 4: Skip after streak of losses
    print("\n  Rule 4: Stop-after-N-losses (0-for-N rule)")
    for n_stop in [1, 2, 3]:
        sub_keep = filt[filt["streak_losses"] < n_stop]
        sub_skip = filt[filt["streak_losses"] >= n_stop]
        m_keep = compute_metrics(sub_keep["outcome_r"].values)
        m_skip = compute_metrics(sub_skip["outcome_r"].values)
        print(f"    0-for-{n_stop}: keep={m_keep['count']:>4} avgR={m_keep['avg_r']:>+.4f} totalR={m_keep['total_r']:>+.2f} "
              f"PPDD={m_keep['ppdd']:.3f} PF={m_keep['pf']:.3f} | "
              f"skip={m_skip['count']:>4} avgR={m_skip['avg_r']:>+.4f}")

    # ─── Now on ALL signals ───
    print("\n\n  B. SEQUENTIAL RULE CANDIDATES (on ALL signals):")

    # Same analysis on all signals
    print("\n  Rule 4b: Stop-after-N-losses (0-for-N rule) — ALL signals:")
    for n_stop in [1, 2, 3, 4]:
        sub_keep = df[df["streak_losses"] < n_stop]
        sub_skip = df[df["streak_losses"] >= n_stop]
        m_keep = compute_metrics(sub_keep["outcome_r"].values)
        m_skip = compute_metrics(sub_skip["outcome_r"].values)
        print(f"    0-for-{n_stop}: keep={m_keep['count']:>4} avgR={m_keep['avg_r']:>+.4f} "
              f"totalR={m_keep['total_r']:>+.2f} | "
              f"skip={m_skip['count']:>4} avgR={m_skip['avg_r']:>+.4f} totalR={m_skip['total_r']:>+.2f}")


def rolling_wr_analysis(df: pd.DataFrame):
    """Does recent rolling win rate predict next signal?"""
    section("BONUS: ROLLING WIN RATE ANALYSIS")

    valid = df.dropna(subset=["rolling_wr_5"])
    print("  Rolling win rate (last 5 signals) vs next outcome:")
    groups = {
        "wr_0%": valid[valid["rolling_wr_5"] == 0],
        "wr_20%": valid[(valid["rolling_wr_5"] > 0) & (valid["rolling_wr_5"] <= 0.2)],
        "wr_40%": valid[(valid["rolling_wr_5"] > 0.2) & (valid["rolling_wr_5"] <= 0.4)],
        "wr_60%": valid[(valid["rolling_wr_5"] > 0.4) & (valid["rolling_wr_5"] <= 0.6)],
        "wr_80%": valid[(valid["rolling_wr_5"] > 0.6) & (valid["rolling_wr_5"] <= 0.8)],
        "wr_100%": valid[valid["rolling_wr_5"] > 0.8],
    }
    groups = {k: v for k, v in groups.items() if len(v) >= 20}
    tbl = metrics_table(groups)
    print(tbl.to_string(index=False))

    # Correlation
    r, p = spearmanr(valid["rolling_wr_5"], valid["outcome_r"])
    print(f"\n  Spearman corr(rolling_wr_5, outcome_r): r={r:.4f}, p={p:.4f}")

    valid2 = df.dropna(subset=["rolling_avg_r_5"])
    r2, p2 = spearmanr(valid2["rolling_avg_r_5"], valid2["outcome_r"])
    print(f"  Spearman corr(rolling_avg_r_5, outcome_r): r={r2:.4f}, p={p2:.4f}")


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    print("=" * 80)
    print("  SEQUENTIAL / CONTEXTUAL FEATURES ANALYSIS")
    print("  NQ Signal Feature Database")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    print(f"\n  Loaded {len(df)} signals from {DATA_PATH.name}")
    print(f"  Date range: {df['bar_time_et'].min()} to {df['bar_time_et'].max()}")
    print(f"  Passes all filters: {df['passes_all_filters'].sum()}")

    # Build sequential features
    print("\n  Building sequential features...")
    df = build_sequential_features(df)

    seq_features = [
        "prev_1_outcome_r", "prev_2_outcome_r", "prev_1_win", "prev_1_loss",
        "streak_wins", "streak_losses",
        "daily_signal_number", "daily_cumulative_r",
        "time_since_last_signal", "time_since_last_win",
        "same_direction_streak", "session_signal_number",
        "rolling_wr_5", "rolling_wr_10", "rolling_avg_r_5",
    ]

    # Quick sanity: show sequential feature stats
    print(f"\n  Sequential feature summary:")
    for f in seq_features:
        valid = df[f].dropna()
        print(f"    {f:<30} mean={valid.mean():>+8.3f}  std={valid.std():>8.3f}  "
              f"min={valid.min():>8.2f}  max={valid.max():>8.2f}  N={len(valid)}")

    # ─── Run analyses ───
    # 1. Correlation
    correlation_analysis(df, seq_features)

    # 2. Conditional performance — all signals
    conditional_performance(df, label="ALL SIGNALS")

    # 3. Hot hand / gambler's fallacy
    hot_hand_analysis(df, label="ALL SIGNALS")

    # 4. Daily pattern
    daily_pattern_analysis(df, label="ALL SIGNALS")

    # 5. XGBoost walk-forward
    xgboost_analysis(df, seq_features)

    # 6. Filtered signals analysis
    filtered_analysis(df)

    # 7. Practical recommendations
    practical_recommendations(df)

    # Bonus
    rolling_wr_analysis(df)

    # ─── FINAL SUMMARY ───
    section("FINAL SUMMARY")
    print("  Key questions answered:")
    print("  1. Is there autocorrelation in outcomes? → see Section 3")
    print("  2. Does the previous signal's outcome predict the next? → see Section 2A")
    print("  3. Is the 1st signal of the day better? → see Section 4A")
    print("  4. Does daily P&L affect later signals (tilt)? → see Section 4B")
    print("  5. Is 0-for-2 optimal? → see Section 4C/D")
    print("  6. Do sequential features improve XGBoost? → see Section 5")
    print("  7. Any actionable new rules? → see Section 7")


if __name__ == "__main__":
    main()
