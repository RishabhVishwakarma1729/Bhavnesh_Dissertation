import numpy as np
import pandas as pd
import pytest

from hrscreen.survey import analyze, cronbach_alpha, make_synthetic_responses, standardise_responses
from hrscreen.survey.analysis import reverse_coded, score_constructs
from hrscreen.survey.questionnaire import CONTEXT, ITEM_IDS, ITEMS, REVERSE_IDS, to_markdown


def test_questionnaire_is_consistent():
    assert len(set(ITEM_IDS)) == len(ITEM_IDS) == len({i.text for i in ITEMS})   # unique ids and unique titles
    assert REVERSE_IDS == ("TR3",)
    md = to_markdown()
    assert all(i.text in md for i in ITEMS) and all(q.text in md for q in CONTEXT)


def test_cronbach_alpha_known_values():
    perfect = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]})
    assert cronbach_alpha(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=(500, 4)))
    assert abs(cronbach_alpha(noise)) < 0.15       # unrelated items are not a scale
    assert np.isnan(cronbach_alpha(perfect.iloc[:2]))


def test_reverse_item_is_flipped_before_scoring():
    df = make_synthetic_responses(5, seed=1)
    df["TR3"] = 1
    assert (reverse_coded(df)["TR3"] == 5).all()


def test_construct_needs_half_the_items_answered():
    df = make_synthetic_responses(3, seed=2)
    df.loc[0, ["PE1", "PE2", "PE3"]] = np.nan
    scored = score_constructs(df)
    assert np.isnan(scored.loc[0, "PE"]) and not np.isnan(scored.loc[1, "PE"])


def test_google_forms_style_headers_and_label_answers_are_understood():
    raw = make_synthetic_responses(20, seed=3, form_style=True)
    labels = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
    first_item_col = ITEMS[0].text
    raw[first_item_col] = raw[first_item_col].map(labels)
    clean, warnings = standardise_responses(raw)
    assert clean["PE1"].between(1, 5).all() and clean["exposure_code"].between(1, 5).all()
    assert warnings == []


def test_missing_item_column_is_an_error():
    raw = make_synthetic_responses(10, seed=4).drop(columns=["PF2"])
    with pytest.raises(ValueError, match="not found"):
        standardise_responses(raw)


def test_analysis_recovers_the_planted_relationships():
    clean, warns = standardise_responses(make_synthetic_responses(300, seed=5, form_style=True))
    result = analyze(clean, warns)
    reg = result.regressions["efficiency_vs_ai_exposure"]["table"].set_index("term")
    assert reg.loc["exposure_code", "coef"] > 0 and reg.loc["exposure_code", "p_value"] < 0.01
    drivers = result.regressions["adoption_intention_drivers"]["table"].set_index("term")
    assert drivers.loc["PE", "coef"] > 0 and drivers.loc["AC", "coef"] < 0
    assert (result.descriptives["cronbach_alpha"].dropna() > 0.5).all()
    assert result.n == 300


def test_synthetic_data_is_loudly_labelled_in_the_report(tmp_path):
    clean, _ = standardise_responses(make_synthetic_responses(70, seed=6))
    result = analyze(clean)
    assert result.synthetic
    assert "SYNTHETIC DATA" in result.to_markdown()
    out = result.save(tmp_path)
    assert (out / "survey_constructs.png").stat().st_size > 5_000


def test_small_samples_trigger_warnings():
    clean, _ = standardise_responses(make_synthetic_responses(25, seed=7))
    text = " ".join(analyze(clean).warnings)
    assert "Only 25 responses" in text and "exploratory" in text
