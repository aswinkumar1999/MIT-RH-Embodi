from flask import Flask, request, jsonify
from flask_cors import CORS  # Import flask-cors

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins and routes

# Global variable to store the last POST request data
last_post_data = None

@app.route('/', methods=['GET', 'POST'])
def home():
    global last_post_data  # Declare the global variable
    if request.method == 'POST':
        # Save the POST data to the global variable
        last_post_data = request.json
        return jsonify({"message": "POST request received", "data": last_post_data}), 200
    if request.method == 'GET':
        # Return the last POST data if it exists, else indicate no data available
        if last_post_data:
            return jsonify({"message": "Last POST request data", "data": last_post_data}), 200
        else:
            return jsonify({"message": "No POST data available"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)