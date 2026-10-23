"""Django settings for the test suite: throwaway data folder and an in-memory database."""
import os
import tempfile
from pathlib import Path

os.environ["HRSCREEN_WEB_DATA"] = tempfile.mkdtemp(prefix="hrscreen-web-test-")
os.environ["HRSCREEN_RESULTS"] = str(Path(__file__).resolve().parents[1] / "results")

from hrscreen.web.settings import *  # noqa: E402,F401,F403

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
DEBUG = True
