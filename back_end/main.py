from digit_prediction import predict_digit
#---------
from flask import Flask, request, jsonify
from flask_cors import CORS  # NEW

app = Flask(__name__)
CORS(app)

@app.route("/digit-predict", methods=["POST", "OPTIONS"])
def handle_digit():
    if request.method == "OPTIONS":
        return '', 200
    try:
        data = request.get_json()
        image = data.get("image")
        result = predict_digit(image)
        return jsonify(result)
    except Exception as e:
        print("Error in /digit-predict:", e)
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)