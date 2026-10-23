"""Django settings for the hrscreen web UI (a local, single-user tool).

Everything configurable is an environment variable, so the same code runs on a laptop, in tests
and behind a server:

* ``HRSCREEN_HOME``      project folder (default: the current directory)
* ``HRSCREEN_WEB_DATA``  where the SQLite database, secret key and survey files live (default: <home>/data/web)
* ``HRSCREEN_RESULTS``   folder of saved benchmark / fairness results (default: <home>/results)
* ``HRSCREEN_DEBUG=1``   show Django error pages with tracebacks
* ``DJANGO_SECRET_KEY``  override the auto-generated secret key
"""

import os
import secrets
from pathlib import Path

HOME = Path(os.environ.get("HRSCREEN_HOME", Path.cwd()))
DATA_DIR = Path(os.environ.get("HRSCREEN_WEB_DATA", HOME / "data" / "web"))
RESULTS_DIR = Path(os.environ.get("HRSCREEN_RESULTS", HOME / "results"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _secret_key() -> str:
    """Use DJANGO_SECRET_KEY if given, otherwise create one once and keep it next to the database."""
    if os.environ.get("DJANGO_SECRET_KEY"):
        return os.environ["DJANGO_SECRET_KEY"]
    key_file = DATA_DIR / "secret_key.txt"
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(50), encoding="utf-8")
    return key_file.read_text(encoding="utf-8").strip()


SECRET_KEY = _secret_key()
DEBUG = os.environ.get("HRSCREEN_DEBUG") == "1"
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]", "testserver", *filter(None, os.environ.get("HRSCREEN_ALLOWED_HOSTS", "").split(","))]

INSTALLED_APPS = ["django.contrib.messages", "hrscreen.web"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
MESSAGE_STORAGE = "django.contrib.messages.storage.cookie.CookieStorage"  # no sessions/accounts needed
MESSAGE_TAGS = {25: "ok", 20: "info", 30: "warn", 40: "bad"}  # success, info, warning, error -> CSS classes
ROOT_URLCONF = "hrscreen.web.urls"
WSGI_APPLICATION = "hrscreen.web.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.messages.context_processors.messages",
    ]},
}]

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": DATA_DIR / "db.sqlite3"}}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

USE_TZ = True
TIME_ZONE = os.environ.get("HRSCREEN_TIME_ZONE", "Asia/Kolkata")
LANGUAGE_CODE = "en"

# ---- upload limits (a resume is a few hundred KB; these are generous but bounded)
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # keep uploads in memory, never as temp files on disk
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 400
MAX_RESUME_BYTES = 5 * 1024 * 1024
MAX_RESUMES = 300
MAX_EMBEDDING_RESUMES = 100   # embedding runs synchronously on the CPU; cap it so a request cannot hang for minutes
MAX_SURVEY_BYTES = 5 * 1024 * 1024
