import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client

load_dotenv()

HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION = os.getenv("HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION")
FRONTEND_DOMAIN_APP = os.getenv("DOMAIN_APP", "http://localhost:3000")
HEADER_NAME = os.getenv("HEADER_NAME")
HEADER_VALUE = os.getenv("HEADER_VALUE")

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", FRONTEND_DOMAIN_APP])

client = Client(HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        header = request.headers.get(HEADER_NAME)
    
        if not header :
            return jsonify({"message": "Need Header to get service"}, 400)
        
        if header != HEADER_VALUE:
            return jsonify({"message": "Wrong header value"}, 400)

        data = request.get_json()
        input_value = data.get("input")

        if input_value is None:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        result = client.predict(input_value, api_name="/predict")

        return jsonify({"message": "success to make a result", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return jsonify({"message": "Archangle AI API service is Ready"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
