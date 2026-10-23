"""Analysis of the HR perception survey (proposal objective 3).

Pipeline: load the Google Forms export -> map columns to items -> reverse-code -> build one score
per construct -> reliability (Cronbach's alpha) -> descriptives -> Spearman correlations ->
OLS regressions -> group comparisons -> figures + a markdown report.

Likert answers are ordinal, so correlations and group tests use rank-based methods. Regressions
treat the 1-5 scores as numeric (standard practice for scale scores) and use heteroskedasticity-
robust (HC3) standard errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..evaluation.benchmark import df_to_markdown
from .questionnaire import CONSTRUCTS, CONTEXT, ITEM_IDS, ITEMS, LIKERT_LABELS, REVERSE_IDS, items_of

CONSTRUCT_IDS = list(CONSTRUCTS)
CODE_COLUMNS = ["exposure_code", "experience_code", "org_size_code"]
_LABEL_TO_NUM = {label.lower(): i + 1 for i, label in enumerate(LIKERT_LABELS)}
TARGET_RESPONSES = (60, 100)  # the proposal's target range of valid responses


# ------------------------------------------------------------------ loading

def standardise_responses(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map exported columns (question text OR item id) to item ids; coerce answers to numbers.

    Returns (clean frame, warnings). Raises if any questionnaire item is missing, because a silent
    gap would quietly change what a construct score means.
    """
    mapping = {i.text.strip(): i.id for i in ITEMS} | {i.id: i.id for i in ITEMS}
    mapping |= {q.text.strip(): q.id for q in CONTEXT} | {q.id: q.id for q in CONTEXT}
    renamed = {c: mapping[str(c).strip()] for c in raw.columns if str(c).strip() in mapping}
    df = raw.rename(columns=renamed)
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [i for i in ITEM_IDS if i not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} questionnaire items not found in the file (e.g. {missing[:3]}). "
            "Question titles in the form must match the text from `hrscreen survey-form` exactly."
        )
    warnings: list[str] = []
    out = pd.DataFrame(index=df.index)
    for item_id in ITEM_IDS:
        numeric = pd.to_numeric(df[item_id], errors="coerce")
        from_label = df[item_id].astype(str).str.strip().str.lower().map(_LABEL_TO_NUM)
        values = numeric.fillna(from_label)
        values = values.where(values.between(1, 5))
        if values.isna().sum() > df[item_id].isna().sum():
            warnings.append(f"{item_id}: some answers were not a 1-5 score or a Likert label and were treated as missing.")
        out[item_id] = values
    for q in CONTEXT:
        if q.id not in df.columns:
            warnings.append(f"Context column '{q.id}' not found; analyses that need it are skipped.")
            continue
        lookup = {o.lower(): n + 1 for n, o in enumerate(q.options)}
        text = df[q.id].astype(str).str.strip()
        out[q.id] = text
        out[f"{q.id}_code"] = text.str.lower().map(lookup)
        if out[f"{q.id}_code"].isna().any():
            warnings.append(f"'{q.id}': some answers do not match the option list and were treated as missing.")
    if "is_synthetic" in df.columns:
        out["is_synthetic"] = df["is_synthetic"].astype(bool)
    return out.reset_index(drop=True), warnings


def load_form_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    return standardise_responses(pd.read_csv(path))


# ------------------------------------------------------------------ scoring and reliability

def reverse_coded(df: pd.DataFrame) -> pd.DataFrame:
    """Items as numbers where HIGHER always means MORE of the construct (reverse items flipped)."""
    items = df[list(ITEM_IDS)].copy()
    for rid in REVERSE_IDS:
        items[rid] = 6 - items[rid]
    return items


def score_constructs(df: pd.DataFrame) -> pd.DataFrame:
    """Add one mean score per construct; needs at least half of a construct's items answered."""
    items = reverse_coded(df)
    out = df.copy()
    for cid in CONSTRUCT_IDS:
        cols = [i.id for i in items_of(cid)]
        out[cid] = items[cols].mean(axis=1, skipna=True).where(items[cols].notna().sum(axis=1) >= math.ceil(len(cols) / 2))
    return out


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Internal consistency of a scale (rule of thumb: >= 0.7 is acceptable). NaN if too little data."""
    x = items.dropna()
    k = x.shape[1]
    if len(x) < 3 or k < 2:
        return float("nan")
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / total_var))


# ------------------------------------------------------------------ analyses

def describe(df: pd.DataFrame) -> pd.DataFrame:
    items = reverse_coded(df)
    rows = []
    for cid, name in CONSTRUCTS.items():
        cols = [i.id for i in items_of(cid)]
        s = df[cid].dropna()
        ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s)) if len(s) > 2 and s.std() > 0 else (np.nan, np.nan)
        rows.append({"construct": cid, "name": name, "items": len(cols), "n": len(s), "mean": s.mean(), "sd": s.std(),
                     "median": s.median(), "ci95_low": ci[0], "ci95_high": ci[1], "cronbach_alpha": cronbach_alpha(items[cols])})
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman rank correlations (pairwise complete). Returns (long table, rho matrix)."""
    cols = [c for c in CODE_COLUMNS if c in df.columns] + CONSTRUCT_IDS
    rho = pd.DataFrame(np.nan, index=cols, columns=cols)
    rows = []
    for a in cols:
        for b in cols:
            pair = df[[a, b]].dropna()
            if a == b:
                rho.loc[a, b] = 1.0
            elif len(pair) > 3 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                r, p = stats.spearmanr(pair[a], pair[b])
                rho.loc[a, b] = r
                if a < b:
                    rows.append({"var_1": a, "var_2": b, "spearman_rho": r, "p_value": p, "n": len(pair)})
    long = pd.DataFrame(rows)
    if len(long):
        long["p_holm"] = _holm(long["p_value"].to_numpy())
        long = long.sort_values("p_value", ignore_index=True)
    return long, rho


def _holm(p: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    return multipletests(p, method="holm")[1]


REGRESSIONS = {
    "efficiency_vs_ai_exposure": ("PE ~ exposure_code + experience_code + org_size_code",
                                  "Perceived efficiency explained by AI-tool exposure (controls: experience, organisation size)"),
    "adoption_intention_drivers": ("AI ~ PE + TR + AC + PF + exposure_code",
                                   "What drives the intention to adopt AI screening"),
    "decision_quality_drivers": ("DQ ~ PE + TR + PF + exposure_code",
                                 "What goes with perceived decision quality"),
}


def regressions(df: pd.DataFrame) -> tuple[dict, list[str]]:
    import statsmodels.formula.api as smf

    out, warnings = {}, []
    for name, (formula, description) in REGRESSIONS.items():
        needed = [t.strip() for t in formula.replace("~", "+").split("+")]
        if any(c not in df.columns for c in needed):
            warnings.append(f"Regression '{name}' skipped: missing columns.")
            continue
        data = df[needed].dropna()
        n_pred = len(needed) - 1
        if len(data) < n_pred + 5:
            warnings.append(f"Regression '{name}' skipped: only {len(data)} complete cases.")
            continue
        if len(data) < 10 * n_pred:
            warnings.append(f"Regression '{name}': n={len(data)} is below the usual 10 cases per predictor ({10 * n_pred}); treat as exploratory.")
        fit = smf.ols(formula, data=data).fit(cov_type="HC3")
        ci = fit.conf_int()
        table = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "robust_se": fit.bse.values,
                              "p_value": fit.pvalues.values, "ci95_low": ci[0].values, "ci95_high": ci[1].values})
        out[name] = {"formula": formula, "description": description, "n": int(fit.nobs),
                     "r2": float(fit.rsquared), "adj_r2": float(fit.rsquared_adj), "table": table}
    return out, warnings


def group_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """AI-tool users vs non-users (Mann-Whitney) and construct differences by role / org size (Kruskal-Wallis)."""
    rows, warnings = [], []
    if "exposure_code" in df.columns:
        users = df["exposure_code"] >= 2
        for cid in CONSTRUCT_IDS:
            a, b = df.loc[users, cid].dropna(), df.loc[~users & df["exposure_code"].notna(), cid].dropna()
            if len(a) < 5 or len(b) < 5:
                warnings.append(f"Users vs non-users on {cid} skipped: a group has fewer than 5 respondents.")
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"comparison": "uses AI tools (>= Rarely) vs never", "construct": cid, "group_sizes": f"{len(a)} vs {len(b)}",
                         "median_a": a.median(), "median_b": b.median(), "effect_rank_biserial": 1 - 2 * u / (len(a) * len(b)), "p_value": p})
    for factor in ("role", "org_size"):
        if factor not in df.columns:
            continue
        for cid in CONSTRUCT_IDS:
            groups = [g[cid].dropna() for _, g in df.groupby(factor) if g[cid].notna().sum() >= 5]
            if len(groups) < 2:
                continue
            h, p = stats.kruskal(*groups)
            rows.append({"comparison": f"by {factor}", "construct": cid, "group_sizes": "/".join(str(len(g)) for g in groups),
                         "median_a": np.nan, "median_b": np.nan, "effect_rank_biserial": np.nan, "p_value": p})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = _holm(table["p_value"].to_numpy())
    return table, warnings


# ------------------------------------------------------------------ orchestration

@dataclass
class SurveyResult:
    data: pd.DataFrame
    n: int
    synthetic: bool
    descriptives: pd.DataFrame
    correlation_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    regressions: dict
    group_tests: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        parts = ["# HR perception survey - analysis"]
        if self.synthetic:
            parts += ["", "> **SYNTHETIC DATA.** These responses were generated by `hrscreen survey-synth` to test the "
                      "pipeline. Relationships were planted on purpose. Do NOT report any of these numbers as findings."]
        lo, hi = TARGET_RESPONSES
        parts += ["", f"- Valid responses: **{self.n}** (target {lo}-{hi})"]
        parts += ["", "## Scales (1-5; reverse-worded items flipped)", df_to_markdown(self.descriptives.drop(columns=["name"]).round(3)),
                  "", "Cronbach's alpha of 0.7 or more is conventionally acceptable.",
                  "", "## Spearman correlations (Holm-adjusted p)",
                  df_to_markdown(self.correlation_table.round(4)) if len(self.correlation_table) else "_not enough data_"]
        for name, r in self.regressions.items():
            parts += ["", f"## Regression: {r['description']}", f"`{r['formula']}` - n = {r['n']}, R2 = {r['r2']:.3f}, adjusted R2 = {r['adj_r2']:.3f} (HC3 robust SEs)",
                      df_to_markdown(r["table"].round(4))]
        parts += ["", "## Group comparisons (Holm-adjusted p)",
                  df_to_markdown(self.group_tests.round(4)) if len(self.group_tests) else "_not enough data_"]
        if self.warnings:
            parts += ["", "## Warnings"] + [f"- {w}" for w in self.warnings]
        parts += ["", "Figures: survey_constructs.png, survey_correlations.png, survey_exposure_vs_efficiency.png"]
        return "\n".join(parts) + "\n"

    def save(self, outdir: str | Path) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(out / "survey_scored.csv", index=False)
        self.descriptives.to_csv(out / "survey_descriptives.csv", index=False)
        self.correlation_table.to_csv(out / "survey_correlations.csv", index=False)
        self.group_tests.to_csv(out / "survey_group_tests.csv", index=False)
        for name, r in self.regressions.items():
            r["table"].to_csv(out / f"survey_regression_{name}.csv", index=False)
        make_figures(self, out)
        (out / "survey_report.md").write_text(self.to_markdown(), encoding="utf-8")
        return out


def analyze(df_clean: pd.DataFrame, extra_warnings: list[str] | None = None) -> SurveyResult:
    """Run every analysis on a standardised response frame (see ``standardise_responses``)."""
    scored = score_constructs(df_clean)
    warnings = list(extra_warnings or [])
    lo, hi = TARGET_RESPONSES
    if len(scored) < lo:
        warnings.append(f"Only {len(scored)} responses; the plan targets {lo}-{hi}. With fewer than ~30 most tests have very low power.")
    corr_long, corr_matrix = correlations(scored)
    reg, reg_warn = regressions(scored)
    groups, grp_warn = group_tests(scored)
    synthetic = bool(scored["is_synthetic"].all()) if "is_synthetic" in scored.columns else False
    return SurveyResult(scored, len(scored), synthetic, describe(scored), corr_long, corr_matrix, reg, groups, warnings + reg_warn + grp_warn)


# ------------------------------------------------------------------ figures

def make_figures(result: SurveyResult, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    watermark = "SYNTHETIC DATA - NOT A FINDING" if result.synthetic else None
    ink, accent, neutral = "#1f2937", "#2563eb", "#9ca3af"

    def finish(fig, ax, name):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if watermark:
            fig.text(0.5, 0.5, watermark, ha="center", va="center", fontsize=22, color="#dc2626", alpha=0.25, rotation=20)
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=160)
        plt.close(fig)

    # 1) construct means with 95% CI
    d = result.descriptives.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvline(3, color=neutral, linestyle="--", linewidth=1)
    ax.errorbar(d["mean"], d["construct"] + "  " + d["name"].str.replace(r" \(.*\)", "", regex=True),
                xerr=[d["mean"] - d["ci95_low"], d["ci95_high"] - d["mean"]], fmt="o", color=accent, ecolor=accent, capsize=3)
    ax.set_xlim(1, 5)
    ax.set_xlabel("Mean score (1 = strongly disagree, 5 = strongly agree), 95% CI; dashed = neutral", color=ink)
    ax.set_title(f"Survey scales (n = {result.n})", loc="left", color=ink)
    finish(fig, ax, "survey_constructs.png")

    # 2) correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(result.correlation_matrix.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Spearman rho"}, linewidths=0.5, linecolor="white")
    ax.set_title("Spearman correlations", loc="left", color=ink)
    finish(fig, ax, "survey_correlations.png")

    # 3) exposure vs perceived efficiency
    data = result.data[["exposure_code", "PE"]].dropna() if "exposure_code" in result.data.columns else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if len(data) > 3:
        rng = np.random.default_rng(0)
        ax.scatter(data["exposure_code"] + rng.normal(0, 0.06, len(data)), data["PE"], s=22, color=accent, alpha=0.6)
        slope, intercept = np.polyfit(data["exposure_code"], data["PE"], 1)
        xs = np.array([1, 5])
        ax.plot(xs, intercept + slope * xs, color=ink, linewidth=1.5)
        ax.set_xticks(range(1, len(CONTEXT[3].options) + 1), CONTEXT[3].options, rotation=20)
    ax.set_ylabel("Perceived efficiency (1-5)", color=ink)
    ax.set_xlabel("How often AI tools are used in hiring", color=ink)
    ax.set_title("AI-tool exposure vs perceived efficiency", loc="left", color=ink)
    finish(fig, ax, "survey_exposure_vs_efficiency.png")
