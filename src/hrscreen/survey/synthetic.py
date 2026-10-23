"""Synthetic survey responses, ONLY for testing the analysis pipeline.

The generator plants known relationships (more AI exposure -> higher perceived efficiency, more
concern -> lower intention, ...) so tests can check that the analysis recovers them. Because the
effects are put in by hand, nothing computed from this data is a research finding. Every row is
marked ``is_synthetic = True`` and the analysis report prints a warning when it sees that flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .questionnaire import CONTEXT, ITEMS


def make_synthetic_responses(n: int = 80, seed: int = 0, form_style: bool = False) -> pd.DataFrame:
    """``n`` fake respondents. ``form_style`` uses question text as column headers, like a Google Forms export."""
    rng = np.random.default_rng(seed)
    ctx = {q.id: rng.choice(q.options, size=n) for q in CONTEXT}
    ctx["exposure"] = rng.choice(CONTEXT[3].options, size=n, p=[0.15, 0.20, 0.25, 0.25, 0.15])
    e = np.array([CONTEXT[3].options.index(v) + 1 for v in ctx["exposure"]]) - 3.0   # centred exposure

    # Latent attitudes with planted effects (the "ground truth" the analysis should recover).
    pe = 3.3 + 0.30 * e + rng.normal(0, 0.5, n)
    pf = 3.0 + 0.15 * e + rng.normal(0, 0.6, n)
    ac = 3.4 - 0.10 * e + rng.normal(0, 0.6, n)
    tr = 3.0 + 0.45 * (pf - 3.0) - 0.25 * (ac - 3.4) + 0.10 * e + rng.normal(0, 0.4, n)
    dq = 3.1 + 0.40 * (pe - 3.3) + 0.30 * (tr - 3.0) + rng.normal(0, 0.4, n)
    ai = 3.2 + 0.40 * (pe - 3.3) + 0.30 * (tr - 3.0) - 0.20 * (ac - 3.4) + rng.normal(0, 0.4, n)
    latent = {"PE": pe, "TR": tr, "AC": ac, "PF": pf, "DQ": dq, "AI": ai}

    data = {k: v for k, v in ctx.items()}
    for item in ITEMS:
        raw = np.clip(np.rint(latent[item.construct] + rng.normal(0, 0.6, n)), 1, 5)
        data[item.id] = (6 - raw if item.reverse else raw).astype(int)
    df = pd.DataFrame(data)
    df["is_synthetic"] = True
    if form_style:
        rename = {i.id: i.text for i in ITEMS} | {q.id: q.text for q in CONTEXT}
        df = df.rename(columns=rename)
        df.insert(0, "Timestamp", pd.date_range("2026-09-01", periods=n, freq="37min").astype(str))
    return df
