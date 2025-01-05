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
citations_filter = re.compile("\[(\d+)\]")

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

            # response = verification_pipeline(user_message)
            
            #HTML Transformation
            atomic_bullets = ""
            
            current_citation = 0
            seen_sources     = dict() # source -> id
            for atomic in response['atomics']:
                # get citations ids
                #if atomic['consensus'] != "no evidence":
                support = atomic['consensus'].replace(" ", "-")
                if support == "support":
                    support = '<span class="material-symbols-outlined" data-icon="check"></span>'
                elif support == "refute":    
                    support = '<span class="material-symbols-outlined" data-icon="close"></span>'
                else: 
                    support = '<span class="material-symbols-outlined" data-icon="warning"></span>'

                # get citations
                citations = citations_filter.findall(atomic['response'])
                citation_ids = [int(cite) - 1 for cite in citations]
                
                source_map = {i: i + 1 for i in citation_ids}
                sources = atomic.get("sources", [])
                for cite in citation_ids:
                    source = sources[cite]['evidence']
                    source = source_filter.match(source).group(1)

                    if source not in seen_sources:
                        seen_sources[source] = current_citation
                        current_citation += 1

                    source_map[cite] = seen_sources[source]

                # change number in text
                def replace_citations(match):
                    original_number = int(match.group(1))  # Extract the original number from the match
                    return f"[{source_map[original_number - 1] + 1}]"  # Map to the new number and return formatted string

                # Apply the replacement
                atomic['response'] = re.sub(r"\[(\d+)\]", replace_citations, atomic['response'])

                # get sources
                atomic_bullets += f"""<li><span class="fw-bold">{support}{atomic['atomic']}</span><p>{atomic['response']}</p></li>\n"""

            # Add sources section
            source_details = ""
            n_sources = sum([ len(atomic['sources']) for atomic in response['atomics']])
            if n_sources != 0:

                source_details += """<
                <details>
                    <summary class="">See sources: </summary>
                    <ol class="source-list">
                """
                
                source_ids = sorted(list(set(seen_sources.values())))
                sources = [atomic.get('sources', []) for atomic in response['atomics']]
                sources = [source for row in sources for source in row]

                print(sources)
                for sid in source_ids:
                    source = sources[sid]
                    source['evidence'] = source_filter.match(source['evidence']).group(1)
                    source_details += f"""
                        <li>{source['evidence']}<br> <i class="ms-4 fw-bold">{source['article']}</i> </li>
                    """
                source_details += """</ol></details></div>"""

            html_output = f"""
                <div class="ms-1">
                    <p>
                        <strong>AI:  {response['general']} </strong>
                    </p>
                    <ul>
                {atomic_bullets}
                    </ul>
                    {source_details}
                """
            
            if response['general'] == "No evidence. The database does not contain evidence to answer the claim.":
                html_output = f"""
                <div class="ms-1">
                    <p>
                        <strong>AI: </strong> {response['general']}
                    </p>
                """

            return jsonify({"result": html_output})
        
        except Exception as error:
            logger.error(f"verification-pipeline error: {error}")      
            return jsonify({"result": "Something Bad Happened!"})
        