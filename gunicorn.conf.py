"""
gunicorn.conf.py
Production Gunicorn configuration for Render and Linux cloud deployments.
"""

import os

# Render dynamically sets $PORT (defaults to 10000 on Render free tier)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Concurrency settings tailored for Render free/starter instances
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
threads = 4
timeout = 120
keepalive = 5

# Logging to stdout/stderr for Render log capture
accesslog = "-"
errorlog = "-"
loglevel = "info"
