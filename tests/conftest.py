import pytest

from hrscreen.data import load_jobs, make_synthetic_resumes


@pytest.fixture(scope="session")
def jobs():
    return load_jobs()


@pytest.fixture(scope="session")
def resumes():
    return make_synthetic_resumes(300, seed=11)
