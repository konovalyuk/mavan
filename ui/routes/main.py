from flask import Blueprint, render_template_string

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def hello_world():
    return "Hello World!"
    # позже для aic:
    # return render_template("home.html")