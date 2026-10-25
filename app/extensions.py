"""
app/extensions.py
Initializes extensions for Flask: SQLAlchemy, CSRFProtect.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
csrf = CSRFProtect()
