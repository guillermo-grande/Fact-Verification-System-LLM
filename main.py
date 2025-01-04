import logging    
from pprint import pprint

from fact_checker import retrieve_engine, embed_model
from fact_checker.data_loaders import logger as db_logger

from flask import Flask
from flask_app.routes import configure_routes

def main():
    app.run(debug=True)
    #response = retrieve_engine.retrieve("Climate change is a ruse")
    #pprint(response)

# Create the Flask app
app = Flask(__name__,template_folder="flask_app/templates",static_folder='flask_app/static')

# Configure the routes
configure_routes(app)

# Run the app
if __name__ == "__main__": main()
    