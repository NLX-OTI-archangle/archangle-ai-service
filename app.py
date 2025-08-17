import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client

# Load .env
load_dotenv()

HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION = os.getenv("HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION")

# Flask init
app = Flask(__name__)
CORS(app)

# Connect ke Hugging Face Space pakai gradio_client
client = Client(HUGGING_FACE_EXAMPLE_LINEAR_REGRESSION)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_value = data.get("input")

        if input_value is None:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        result = client.predict(input_value, api_name="/predict")

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return jsonify({"message": "Flask + Gradio service running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
