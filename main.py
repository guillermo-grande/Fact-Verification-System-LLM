import logging    
from pprint import pprint
from fact_checker.query_pipeline import verification_pipeline
from fact_checker.query_pipeline import logger as verify_logger

from flask import Flask
from flask_app.routes import configure_routes

def main():
    app.run(debug=True)
    #response = retrieve_engine.retrieve("Climate change is a ruse")
    #pprint(response)a

# Create the Flask app
app = Flask(__name__, template_folder="flask_app/templates", static_folder='flask_app/static')

# Configure the routes
configure_routes(app)

# Run the app
if __name__ == "__main__": main()
    
