"""
run.py
Development server entry point for Web Analytics-Based User Conversion Prediction System.
"""

import os
from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    
    print("\n" + "=" * 65)
    print(" Web Analytics-Based User Conversion Prediction System")
    print(f" Server starting on: http://127.0.0.1:{port}")
    print(f" Dashboard:          http://127.0.0.1:{port}/dashboard")
    print(f" Predict:            http://127.0.0.1:{port}/predict")
    print(f" Evaluation:         http://127.0.0.1:{port}/evaluation")
    print("=" * 65 + "\n")

    app.run(host="127.0.0.1", port=port, debug=debug)
