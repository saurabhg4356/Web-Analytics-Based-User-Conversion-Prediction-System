"""
wsgi.py
Production WSGI entry point for Gunicorn, Waitress, or Cloud Platforms.
"""

import os
from app import create_app

# Resolve configuration (checks FLASK_CONFIG or FLASK_ENV, defaulting to production)
config_name = os.environ.get("FLASK_CONFIG") or os.environ.get("FLASK_ENV") or "production"
application = create_app(config_name)
app = application  # Alias for common WSGI runners (e.g. gunicorn wsgi:app)

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    print(f"[*] Starting production Waitress WSGI server on port {port}...")
    serve(application, host="0.0.0.0", port=port)
