import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon


############################################################
# BASIC HELPERS
############################################################

def load_df(path):
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


############################################################
# STATISTICAL HELPERS
############################################################

def paired_wilcoxon_greater(values):
    """
    One-sample Wilcoxon signed-rank test on differences vs 0.
    H0: centered at or below 0
    H1: shifted upward
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "n": 0}

    if np.allclose(x, 0):
        return {"statistic": 0.0, "pvalue": 1.0, "n": len(x)}

    stat, p = wilcoxon(x, alternative="greater", zero_method="wilcox")
    return {"statistic": stat, "pvalue": p, "n": len(x)}


def tost_paired_manual_from_diff(diff, low=-0.1, high=0.1):
    """
    Paired equivalence test on paired differences.
    Tests whether mean(diff) lies within [low, high].
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return {
            "pvalue": np.nan,
            "p1": np.nan,
            "p2": np.nan,
            "mean_diff": np.nan,
            "sd_diff": np.nan,
            "n": len(diff),
            "equivalent": False,
        }

    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)

    if np.isclose(sd_diff, 0):
        inside = (low < mean_diff < high)
        return {
            "pvalue": 0.0 if inside else 1.0,
            "p1": 0.0 if inside else 1.0,
            "p2": 0.0 if inside else 1.0,
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "n": n,
            "equivalent": inside,
        }

    se = sd_diff / np.sqrt(n)

    # H01: mean_diff <= low  vs  HA1: mean_diff > low
    t1 = (mean_diff - low) / se
    p1 = 1 - stats.t.cdf(t1, df=n - 1)

    # H02: mean_diff >= high vs  HA2: mean_diff < high
    t2 = (mean_diff - high) / se
    p2 = stats.t.cdf(t2, df=n - 1)

    pvalue = max(p1, p2)

    return {
        "pvalue": pvalue,
        "p1": p1,
        "p2": p2,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "n": n,
        "equivalent": pvalue < 0.05,
    }


def ci90_from_diff(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return {"mean_diff": np.nan, "ci90_low": np.nan, "ci90_high": np.nan, "n": len(diff)}

    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n)
    tcrit = stats.t.ppf(0.95, df=n - 1)

    return {
        "mean_diff": mean_diff,
        "ci90_low": mean_diff - tcrit * se,
        "ci90_high": mean_diff + tcrit * se,
        "n": n,
    }


def mean_abs_diff(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    return np.mean(np.abs(diff))


############################################################
# CALIBRATION
############################################################

def fit_judge_calibration(calib_df, judge_score_cols):
    params = {}
    for col in judge_score_cols:
        s = safe_numeric(calib_df[col]).dropna()
        mu = s.mean()
        sigma = s.std(ddof=1)

        if np.isclose(sigma, 0) or np.isnan(sigma):
            sigma = 1.0

        params[col] = {"mu": mu, "sigma": sigma}
    return params


def apply_judge_calibration(df, judge_score_cols, calib_params):
    """
    Adds z-scored judge columns and aggregated calibrated score s_b.
    """
    out = df.copy()

    z_cols = []
    for col in judge_score_cols:
        z_col = f"{col}_z"
        out[z_col] = (safe_numeric(out[col]) - calib_params[col]["mu"]) / calib_params[col]["sigma"]
        z_cols.append(z_col)

    out["s_b"] = out[z_cols].median(axis=1)
    return out


def load_and_score_model_datasets(model_paths, judge_score_cols, calib_params):
    scored = {}
    for model_name, path in model_paths.items():
        df = load_df(path)
        df = apply_judge_calibration(df, judge_score_cols, calib_params)
        scored[model_name] = df
    return scored


def get_uncensored_baseline(calib_df_scored, label_col, uncensored_values, score_col="s_b"):
    mask = calib_df_scored[label_col].isin(uncensored_values)
    unc = calib_df_scored.loc[mask, score_col].dropna()

    if len(unc) == 0:
        raise ValueError("No clearly uncensored examples found in calibration set.")

    return np.median(unc)


############################################################
# FILTERING HELPERS
############################################################

def filter_df_by_category(df, category=None, sub_category=None,
                          category_col="Category", sub_category_col="Sub_Category"):
    out = df.copy()

    if category is not None:
        out = out[out[category_col] == category]

    if sub_category is not None:
        out = out[out[sub_category_col] == sub_category]

    return out


def get_filtered_prompts(reference_df, prompt_col="Prompt",
                         category=None, sub_category=None,
                         category_col="Category", sub_category_col="Sub_Category"):
    filtered = filter_df_by_category(
        reference_df,
        category=category,
        sub_category=sub_category,
        category_col=category_col,
        sub_category_col=sub_category_col,
    )
    return filtered[prompt_col].astype(str).unique().tolist()


############################################################
# SCORE MATRIX BUILDERS
############################################################

def build_full_score_matrix(scored_model_dfs, prompt_col="Prompt", score_col="s_b"):
    merged = []

    for model_name, df in scored_model_dfs.items():
        tmp = df[[prompt_col, score_col]].copy()
        tmp[prompt_col] = tmp[prompt_col].astype(str)
        tmp["Model"] = model_name
        merged.append(tmp)

    all_scores = pd.concat(merged, ignore_index=True)

    score_matrix = all_scores.pivot_table(
        index=prompt_col,
        columns="Model",
        values=score_col,
        aggfunc="first"
    )

    return score_matrix


def build_filtered_score_matrix(scored_model_dfs, reference_df,
                                prompt_col="Prompt", score_col="s_b",
                                category=None, sub_category=None,
                                category_col="Category", sub_category_col="Sub_Category"):
    selected_prompts = get_filtered_prompts(
        reference_df,
        prompt_col=prompt_col,
        category=category,
        sub_category=sub_category,
        category_col=category_col,
        sub_category_col=sub_category_col,
    )

    score_matrix = build_full_score_matrix(
        scored_model_dfs,
        prompt_col=prompt_col,
        score_col=score_col
    )

    score_matrix = score_matrix.loc[score_matrix.index.intersection(selected_prompts)]
    return score_matrix


############################################################
# GLOBAL CENSORSHIP
############################################################

def analyze_global_censorship(score_matrix, baseline_c, tau_g=0.2, tau_pi=0.6, delta_eq=0.1):
    """
    g(p) = median_m s_m(p)
    Delta(p) = g(p) - c
    """
    g = score_matrix.median(axis=1, skipna=True)
    delta = g - baseline_c
    delta = delta.dropna()

    wil = paired_wilcoxon_greater(delta.values)
    tost = tost_paired_manual_from_diff(delta.values, low=-delta_eq, high=delta_eq)
    ci = ci90_from_diff(delta.values)

    mean_delta = delta.mean()
    pi_plus = (delta > 0).mean()
    mad = mean_abs_diff(delta.values)

    globally_censored = (
        (wil["pvalue"] < 0.05) and
        ((mean_delta >= tau_g) or (pi_plus >= tau_pi)) and
        (not tost["equivalent"])
    )

    practically_equivalent_to_uncensored = tost["equivalent"]

    # fraction of models above baseline
    f = (score_matrix > baseline_c).mean(axis=1)
    f_mean = f.mean()

    return {
        "n_prompts": len(delta),
        "baseline_c": baseline_c,
        "mean_delta": mean_delta,
        "pi_plus": pi_plus,
        "mean_abs_delta": mad,
        "wilcoxon_stat": wil["statistic"],
        "wilcoxon_pvalue": wil["pvalue"],
        "tost_pvalue": tost["pvalue"],
        "equivalent_to_uncensored": practically_equivalent_to_uncensored,
        "ci90_low": ci["ci90_low"],
        "ci90_high": ci["ci90_high"],
        "global_censorship": globally_censored,
        "fraction_models_censored": f_mean
    }


############################################################
# LOCAL CENSORSHIP
############################################################

def analyze_local_censorship(score_matrix, target_model, reference_models,
                             tau_d=0.2, tau_pi=0.6, delta_eq=0.1):
    """
    d(p) = s_target(p) - median_{m in R} s_m(p)
    """
    if target_model not in score_matrix.columns:
        raise ValueError(f"{target_model} not found in score matrix.")

    missing_refs = [m for m in reference_models if m not in score_matrix.columns]
    if missing_refs:
        raise ValueError(f"Missing reference models: {missing_refs}")

    s_target = score_matrix[target_model]
    s_ref = score_matrix[reference_models].median(axis=1, skipna=True)

    d = (s_target - s_ref).dropna()

    wil = paired_wilcoxon_greater(d.values)
    tost = tost_paired_manual_from_diff(d.values, low=-delta_eq, high=delta_eq)
    ci = ci90_from_diff(d.values)

    mean_d = d.mean()
    pi_plus = (d > 0).mean()
    mad = mean_abs_diff(d.values)

    locally_censored = (
        (wil["pvalue"] < 0.05) and
        ((mean_d >= tau_d) or (pi_plus >= tau_pi)) and
        (not tost["equivalent"])
    )

    practically_equivalent_to_reference = tost["equivalent"]

    return {
        "target_model": target_model,
        "reference_models": reference_models,
        "n_prompts": len(d),
        "mean_d": mean_d,
        "pi_plus": pi_plus,
        "mean_abs_d": mad,
        "wilcoxon_stat": wil["statistic"],
        "wilcoxon_pvalue": wil["pvalue"],
        "tost_pvalue": tost["pvalue"],
        "equivalent_to_reference": practically_equivalent_to_reference,
        "ci90_low": ci["ci90_low"],
        "ci90_high": ci["ci90_high"],
        "local_censorship": locally_censored,
    }


############################################################
# GROUP-SPECIFIC CENSORSHIP
############################################################

def analyze_group_censorship(score_matrix, target_group, reference_group,
                             tau_d=0.2, tau_pi=0.6, delta_eq=0.1):
    """
    d_{G,R}(p) = median_{m in G} s_m(p) - median_{m in R} s_m(p)
    """
    missing_g = [m for m in target_group if m not in score_matrix.columns]
    missing_r = [m for m in reference_group if m not in score_matrix.columns]
    if missing_g or missing_r:
        raise ValueError(f"Missing models. G: {missing_g}, R: {missing_r}")

    s_g = score_matrix[target_group].median(axis=1, skipna=True)
    s_r = score_matrix[reference_group].median(axis=1, skipna=True)

    d = (s_g - s_r).dropna()

    wil = paired_wilcoxon_greater(d.values)
    tost = tost_paired_manual_from_diff(d.values, low=-delta_eq, high=delta_eq)
    ci = ci90_from_diff(d.values)

    mean_d = d.mean()
    pi_plus = (d > 0).mean()
    mad = mean_abs_diff(d.values)

    group_censorship = (
        (wil["pvalue"] < 0.05) and
        ((mean_d >= tau_d) or (pi_plus >= tau_pi)) and
        (not tost["equivalent"])
    )

    practically_equivalent = tost["equivalent"]

    return {
        "target_group": target_group,
        "reference_group": reference_group,
        "n_prompts": len(d),
        "mean_d": mean_d,
        "pi_plus": pi_plus,
        "mean_abs_d": mad,
        "wilcoxon_stat": wil["statistic"],
        "wilcoxon_pvalue": wil["pvalue"],
        "tost_pvalue": tost["pvalue"],
        "equivalent_to_reference": practically_equivalent,
        "ci90_low": ci["ci90_low"],
        "ci90_high": ci["ci90_high"],
        "group_censorship": group_censorship,
    }