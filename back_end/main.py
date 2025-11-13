from flask import Flask, request, jsonify
from digit_prediction import predict_digit

app = Flask(__name__)

@app.route("/digit-predict", methods=["POST"])
def handle_digit():
    data = request.get_json()  # Expecting JSON data
    image = data.get("image")
    result = predict_digit(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)