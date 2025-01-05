from flask import Flask
from flask import render_template, request, jsonify

# Define the routes
def configure_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    # Route to handle messages (POST request)
    @app.route('/send-message', methods=['POST'])
    def send_message():
        user_message = request.form.get('message')
        print(user_message)
        return jsonify({"result": "Fact verification result here!"})
        
    @app.route('/verify')
    def verify_fact():
        # Example response from some function (REPLACE)
        return jsonify({"result": "Fact verification result here!"})