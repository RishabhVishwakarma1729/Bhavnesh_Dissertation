from datetime import date

from hrscreen.parsing import parse_resume, redact_pii
from hrscreen.parsing.skills import default_lexicon


def skills(text):
    return set(default_lexicon().extract(text))


def test_skills_aliases_and_symbols():
    assert {"machine learning", "kubernetes", "node.js"} <= skills("ML, K8s and NodeJS experience")
    assert "scikit-learn" in skills("used sklearn for modelling")


def test_skills_respect_word_boundaries():
    found = skills("Experienced with MySQL and JavaScript")
    assert "mysql" in found and "javascript" in found
    assert "sql" not in found and "java" not in found


def test_skills_are_case_and_hyphen_insensitive():
    assert "machine learning" in skills("Machine-Learning models")


def test_education_takes_the_highest_level():
    assert parse_resume("B.Tech in CS and M.Tech in AI").education_level == "master"
    assert parse_resume("Ph.D in statistics, MBA").education_level == "phd"
    assert parse_resume("Diploma in civil engineering").education_level == "diploma"
    assert parse_resume("nothing relevant here").education_level == "unknown"


def test_education_works_on_stripped_text():
    assert parse_resume("bachelor science accounting richland college").education_level == "bachelor"


def test_experience_total_statement_wins():
    assert parse_resume("Over 7 years of experience in data engineering").years_experience == 7.0
    assert parse_resume("5+ yrs experience with python").years_experience == 5.0


def test_experience_sums_per_job_durations():
    assert parse_resume("Exprience - 24 months at A. Exprience - 12 months at B.").years_experience == 3.0


def test_experience_from_date_ranges_ignores_education_and_overlaps():
    text = "Education: B.Tech 2008-2012. Worked at Acme 2013 - 2016 and Beta 2015 - 2018."
    # 2013-2018 after merging the overlap; the degree years are skipped.
    assert parse_resume(text).years_experience == 5.0


def test_experience_present_uses_current_year():
    text = "work experience analyst 2020 to present"
    assert parse_resume(text).years_experience == float(date.today().year - 2020)


def test_experience_unknown_is_none():
    assert parse_resume("no dates anywhere").years_experience is None


def test_redact_pii_removes_contacts_but_keeps_year_lists():
    out = redact_pii("mail me at jo@x.com or +91 98765 43210 or 555 589 7312, see https://x.com/jo. worked 2016 2018 2020")
    for leaked in ("jo@x.com", "98765", "589 7312", "https://"):
        assert leaked not in out
    assert "2016 2018 2020" in out
