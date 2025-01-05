from logging import getLogger

from flask import Flask
from flask import render_template, request, jsonify

# from fact_checker.query_pipeline import verification_pipeline
from fact_checker.utils import configure_logger
import json

logger = getLogger('server-side')
logger = configure_logger(logger)

# Define the routes
def configure_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    # Route to handle messages (POST request)
    @app.route('/send-message', methods=['POST'])
    def send_message():
        user_message = request.form.get('message')
        if len(user_message) == 0: 
            return jsonify({"result": "Please don't write gribrish!"})
        try:
            with open('tmp.json') as tmp:
                response = json.load(tmp)
                # response = jsonify(response)

            # llm_response = verification_pipeline(user_message)
            
            # # HTML Transformation
            # html_output = f"""
            #                 <div id="response-div">
            #                     <p id="response-general">
            #                         {llm_response['general']}
            #                     </p>
            #                     <ul>
            #                 """
            # for atomic in llm_response['atomics']:
            #     if atomic['consensus'] != "no evidence":
            #         html_output += f"<li>{atomic['response']}</li>\n"
            # html_output += "</ul>"

            return jsonify({"result": response})
        except Exception as error:
            logger.error(f"verification-pipeline error: {error}")      
            return jsonify({"result": "Something Bad Happened!"})
        