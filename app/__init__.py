"""
app/__init__.py
Flask Application Factory module.
"""

import os
from flask import Flask, render_template
from app.config import config_by_name
from app.extensions import db, csrf
from app.routes import (
    main_bp,
    dashboard_bp,
    prediction_bp,
    history_bp,
    evaluation_bp,
    data_explorer_bp
)


def create_app(config_name: str = None) -> Flask:
    """Application factory for Web Analytics User Conversion Prediction System."""
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development").lower()

    app = Flask(__name__)
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    csrf.init_app(app)

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(data_explorer_bp)

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html", error=e), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template("500.html", error=e), 500

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("404.html", error=e, custom_title="400 Bad Request"), 400

    # Context Processors for Templates
    @app.context_processor
    def inject_global_template_vars():
        return {
            "app_name": "Web Analytics Conversion System",
            "app_version": "1.0.0",
            "model_version": "v1.0.0 (Random Forest)",
            "default_threshold": app.config.get("DEFAULT_DECISION_THRESHOLD", 0.40)
        }

    # Auto-initialize database tables in app context
    with app.app_context():
        db.create_all()

    return app
