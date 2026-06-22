from config import flask_settings
from ui import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=flask_settings.HOST,
        port=flask_settings.PORT,
        debug=flask_settings.DEBUG,
    )