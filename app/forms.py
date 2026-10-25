"""
app/forms.py
Flask-WTF forms for validating session features in the prediction interface.
"""

from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional, Length, InputRequired


class SessionPredictionForm(FlaskForm):
    """Server-side validated form for pre-purchase visitor session conversion estimation."""

    # Browsing activity
    administrative = IntegerField(
        "Administrative Pages",
        validators=[InputRequired(), NumberRange(min=0, max=50, message="Administrative pages must be between 0 and 50.")],
        default=0,
        description="Number of account, profile, or cart admin pages visited."
    )
    administrative_duration = FloatField(
        "Administrative Duration (seconds)",
        validators=[InputRequired(), NumberRange(min=0.0, max=5000.0, message="Duration must be between 0 and 5,000s.")],
        default=0.0,
        description="Total seconds spent on administrative pages."
    )
    informational = IntegerField(
        "Informational Pages",
        validators=[InputRequired(), NumberRange(min=0, max=50, message="Informational pages must be between 0 and 50.")],
        default=0,
        description="Number of about, FAQ, blog, or contact pages visited."
    )
    informational_duration = FloatField(
        "Informational Duration (seconds)",
        validators=[InputRequired(), NumberRange(min=0.0, max=5000.0, message="Duration must be between 0 and 5,000s.")],
        default=0.0,
        description="Total seconds spent on informational pages."
    )
    product_related = IntegerField(
        "Product-Related Pages",
        validators=[InputRequired(), NumberRange(min=0, max=1000, message="Product-related pages must be between 0 and 1,000.")],
        default=12,
        description="Number of product catalog or item detail pages viewed."
    )
    product_related_duration = FloatField(
        "Product Duration (seconds)",
        validators=[InputRequired(), NumberRange(min=0.0, max=50000.0, message="Duration must be between 0 and 50,000s.")],
        default=450.0,
        description="Total seconds spent reviewing product details."
    )

    # Google Analytics Metrics
    bounce_rates = FloatField(
        "Bounce Rate (0.00 - 1.00)",
        validators=[InputRequired(), NumberRange(min=0.0, max=1.0, message="Bounce rate must be between 0.0 and 1.0.")],
        default=0.01,
        description="Percentage of visitors who enter the site and leave without viewing other pages."
    )
    exit_rates = FloatField(
        "Exit Rate (0.00 - 1.00)",
        validators=[InputRequired(), NumberRange(min=0.0, max=1.0, message="Exit rate must be between 0.0 and 1.0.")],
        default=0.03,
        description="Percentage of pageviews that were the last in the visitor session."
    )
    special_day = FloatField(
        "Special Day Proximity (0.00 - 1.00)",
        validators=[InputRequired(), NumberRange(min=0.0, max=1.0, message="Special Day must be between 0.0 and 1.0.")],
        default=0.0,
        description="Closeness of browsing date to a holiday or shopping event (e.g. Black Friday, Mother's Day)."
    )

    # Temporal & Demographic Categorical Attributes
    month = SelectField(
        "Session Month",
        choices=[
            ("Feb", "February"),
            ("Mar", "March"),
            ("May", "May"),
            ("June", "June"),
            ("Jul", "July"),
            ("Aug", "August"),
            ("Sep", "September"),
            ("Oct", "October"),
            ("Nov", "November"),
            ("Dec", "December")
        ],
        default="Nov",
        validators=[DataRequired()]
    )
    operating_systems = SelectField(
        "Operating System",
        choices=[(str(i), f"OS Type {i}") for i in range(1, 9)],
        default="2",
        validators=[DataRequired()]
    )
    browser = SelectField(
        "Browser",
        choices=[(str(i), f"Browser Type {i}") for i in range(1, 14)],
        default="2",
        validators=[DataRequired()]
    )
    region = SelectField(
        "Geographic Region",
        choices=[(str(i), f"Region {i}") for i in range(1, 10)],
        default="1",
        validators=[DataRequired()]
    )
    traffic_type = SelectField(
        "Traffic Source Category",
        choices=[(str(i), f"Traffic Channel {i}") for i in range(1, 21)],
        default="2",
        validators=[DataRequired()]
    )
    visitor_type = SelectField(
        "Visitor Type",
        choices=[
            ("Returning_Visitor", "Returning Visitor"),
            ("New_Visitor", "New Visitor"),
            ("Other", "Other")
        ],
        default="Returning_Visitor",
        validators=[DataRequired()]
    )
    weekend = SelectField(
        "Weekend Session?",
        choices=[
            ("False", "No (Weekday Session)"),
            ("True", "Yes (Weekend Session)")
        ],
        default="False",
        validators=[DataRequired()]
    )

    # Operational Decision Threshold
    decision_threshold = FloatField(
        "Decision Threshold (0.10 - 0.90)",
        validators=[InputRequired(), NumberRange(min=0.05, max=0.95, message="Threshold must be between 0.05 and 0.95.")],
        default=0.40,
        description="Probability cutoff to classify as predicted purchase. Lower threshold favors higher recall."
    )
    session_summary = StringField(
        "Session Label / Campaign Tag",
        validators=[Optional(), Length(max=100)],
        description="Optional tag for tracking in prediction history (e.g., 'Holiday Ad Campaign #4')."
    )

    submit = SubmitField("Predict Purchase Probability")
