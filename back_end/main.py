from digit_prediction import predict_digit
#---------
from flask import Flask, request, jsonify
from flask_cors import CORS  # NEW

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://personal-website-pi-rosy.vercel.app'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route("/digit-predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    try:
        data = request.get_json()
        image = data.get("image")
        result = predict_digit(image)
        return jsonify(result)
    except Exception as e:
        print("Error in /digit-predict:", e)
        return jsonify({"error": str(e)}), 500


