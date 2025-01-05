from logging import getLogger

from flask import Flask
from flask import render_template, request, jsonify

from fact_checker.query_pipeline import verification_pipeline
from fact_checker.utils import configure_logger
import re
import json

logger = getLogger('server-side')
logger = configure_logger(logger)

# regrex pattern
source_filter = re.compile("Source [\d]+:\n(.*)")

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
            # with open('tmp.json') as tmp:
                # response = json.load(tmp)
                # response = jsonify(response)

            response = verification_pipeline(user_message)
            
            #HTML Transformation
            atomic_bullets = ""
            for atomic in response['atomics']:
                # get citations ids
                if atomic['consensus'] != "no evidence":
                    atomic_bullets += f"""<li><span class="fw-bold">{atomic['atomic']}</span><p>{atomic['response']}</p></li>\n"""

            # Add sources section
            source_details = ""
            n_sources = sum([ len(atomic['sources']) for atomic in response['atomics']])
            if n_sources != 0:
                source_details += """
                <details>
                    <summary class="">See sources: </summary>
                    <ol class="source-list">
                """
                
                for atomic in response['atomics']:
                    if atomic['consensus'] != "no evidence":
                        for i, source in enumerate(atomic.get('sources', []), 1):
                            source['evidence'] = source_filter.match(source['evidence']).group(1)
                            source_details += f"""
                                <li>
                                    {source['evidence']}
                                    <br>
                                    <i class="ms-4 fw-bold">{source['article']}</i>
                                </li>
                            """
                source_details += """
                        </ol>
                    </details>
                </div>
                """

            html_output = f"""
                <div class="ms-1">
                    <p>
                        <strong>AI: </strong> {response['general']}
                    </p>
                    <ul>
                {atomic_bullets}
                    </ul>
                    {source_details}
                """
            return jsonify({"result": html_output})
        
        except Exception as error:
            logger.error(f"verification-pipeline error: {error}")      
            return jsonify({"result": "Something Bad Happened!"})
        