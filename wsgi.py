"""
wsgi.py
Production WSGI entry point for Gunicorn, Waitress, or Cloud Platforms.
"""

import os
from app import create_app

# Use production configuration by default in WSGI entry point
application = create_app(os.environ.get("FLASK_ENV", "production"))
app = application  # Alias for common WSGI runners

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    print(f"[*] Starting production Waitress WSGI server on port {port}...")
    serve(application, host="0.0.0.0", port=port)
