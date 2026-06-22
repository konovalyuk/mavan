from ui.routes.main import main_bp

# позже добавите:
# from ui.routes.contracts import contracts_bp
# from ui.routes.custom import custom_bp


def register_blueprints(app):
    app.register_blueprint(main_bp)
    # app.register_blueprint(contracts_bp)
    # app.register_blueprint(custom_bp)