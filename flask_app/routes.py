from logging import getLogger

from flask import Flask
from flask import Blueprint
from flask import render_template, request, jsonify

from fact_checker.query_pipeline import verification_pipeline
from fact_checker.utils import configure_logger
import re
import json

logger = getLogger('server-side')
logger = configure_logger(logger)

# regrex pattern
source_filter = re.compile(r"\[\s*\d+\s*\]\s(.*)")
citations_filter = re.compile(r"\[(\d+)\]")

# Blueprint
route_blueprint = Blueprint("routes", __name__, template_folder='templates')

@route_blueprint.route('/')
def index():
    return render_template('index.html')
 
# Route to handle messages (POST request)
def normalize_citations(response) -> tuple[str, dict[int, int]]:
    atomic_bullets = ""
    
    seen_sources     = dict() # source -> id
    def replace_match(match):
        return f"[{source_map[int(match.group(1)) - 1] + 1}]"

    current_citation = 0
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
                seen_sources[source] = (current_citation, sources[cite]['article'])
                current_citation += 1


            source_map[cite] = seen_sources[source][0]

        # get sources
        atomic['response'] = re.sub(r"\[(\d+)\]", replace_match, atomic['response'])
        atomic_bullets += f"""<li><span class="fw-bold">{support}{atomic['atomic']}</span><p>{atomic['response']}</p></li>\n"""

    return atomic_bullets, seen_sources


@route_blueprint.route('/send-message', methods=['POST'])
def send_message():
    user_message = request.form.get('message')
    if len(user_message) == 0: 
        return jsonify({"result": "Please don't write gribrish!"})
    try:

        with open('tmp.json', 'r') as tmp: response = json.load(tmp)
        response = verification_pipeline(user_message)
        # with open('tmp.json', 'w') as tmp: json.dump(response, tmp, indent = 2)
        #HTML Transformation
        atomic_bullets, seen_sources = normalize_citations(response)

        # Add sources section
        source_details = ""
        
        n_sources = sum([ len(atomic['sources']) for atomic in response['atomics']])
        if n_sources != 0:

            source_details += f"""
            <details>
                <summary class="fw-bold">{response['see-sources']}: </summary>
                <ol class="source-list">
            """

            for evidence, (_, article) in seen_sources.items():
                evidence = evidence.capitalize()
                source_details += f"""
                    <li>{evidence}<br> <i class="ms-4 fw-bold">{article}</i> </li>
                """
            source_details += """</ol></details></div>"""

        html_output = f"""
            <div class="ms-1 d-flex flex-column">
                <p>
                    <strong>AI:  {response['general']} </strong>
                </p>
                <div class="ms-4">
                    <ul>
                        {atomic_bullets}
                    </ul>
                    {source_details}
                </div>
            """
        
        if not response['verified']:
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
    