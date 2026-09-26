import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # makes HF_TOKEN etc. available via os.environ

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h]

INSTALLED_APPS = ["corsheaders", "predictor"]

# No CSRF middleware: this is a stateless JSON API with no cookies/sessions.
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
]

# React dev servers are allowed by default; add your deployed frontend URL in .env
CORS_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
DATABASES = {}  # no database needed
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
