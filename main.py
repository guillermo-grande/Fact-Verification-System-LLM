import logging    
from pprint import pprint

# from fact_checker.query_pipeline import verification_pipeline
# from fact_checker.query_pipeline import logger as verify_logger

from flask import Flask
from flask_app.routes import route_blueprint

def main():
    app.register_blueprint(route_blueprint)
    app.run(debug=False)


# Create the Flask app
app = Flask(__name__, template_folder="flask_app/templates", static_folder='flask_app/static')

# Run the app
if __name__ == "__main__": main()
    
