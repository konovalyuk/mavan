from flask import Flask

from config import flask_settings
from ui.routes import register_blueprints


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = flask_settings.SECRET_KEY

    register_blueprints(app)

    return app
