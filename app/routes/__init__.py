"""
app/routes/__init__.py
Exposes all application route blueprints.
"""

from app.routes.main import main_bp
from app.routes.dashboard import dashboard_bp
from app.routes.prediction import prediction_bp
from app.routes.history import history_bp
from app.routes.evaluation import evaluation_bp
from app.routes.data_explorer import data_explorer_bp

__all__ = [
    "main_bp",
    "dashboard_bp",
    "prediction_bp",
    "history_bp",
    "evaluation_bp",
    "data_explorer_bp"
]
