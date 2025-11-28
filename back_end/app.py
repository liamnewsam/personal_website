import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from w101.Game import Game  # <-- your Game class import

from back_end.digit_prediction.digit_prediction import predict_digit  # your existing code

# -------------------------------------------------------
# Flask Application
# -------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=["https://liamcnewsam.com"])

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",   # VERY IMPORTANT for Cloud Run WebSockets
)

# -------------------------------------------------------
# Existing Flask Route (unchanged)
# -------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://liamcnewsam.com"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
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
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# Simple In-Memory Game State
# -------------------------------------------------------

active_games = {}   # room_id -> Game instance
waiting_players = []  # matchmaking queue

# -------------------------------------------------------
# WebSocket Events
# -------------------------------------------------------

@socketio.on("connect")
def on_connect():
    print("Client connected:", request.sid)
    emit("connected", {"id": request.sid})


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected:", request.sid)
    # cleanup if needed (optional)


@socketio.on("join_queue")
def join_queue():
    """Player wants to enter matchmaking."""
    print("Player joined queue:", request.sid)
    waiting_players.append(request.sid)

    # Simple 1v1 matchmaking; we can expand to 2v2, 4v4 later
    if len(waiting_players) >= 2:
        p1 = waiting_players.pop(0)
        p2 = waiting_players.pop(0)

        room_id = f"room_{p1[:5]}_{p2[:5]}"
        print("Starting match:", room_id)

        # Create your game class instance
        game = Game(players=[p1, p2])
        active_games[room_id] = game

        join_room(room_id, sid=p1)
        join_room(room_id, sid=p2)

        # Notify clients
        socketio.emit("match_start", {"room": room_id}, to=room_id)
        socketio.emit("state_update", game.serialize(), to=room_id)


@socketio.on("cast_spell")
def cast_spell(data):
    """Player submits chosen spell for this round."""
    room_id = data["room"]
    spell = data["spell"]
    target = data.get("target")

    game = active_games.get(room_id)
    if not game:
        return

    # Queue the player's action (your engine handles this)
    game.queue_action(request.sid, spell, target)

    # If everyone has chosen a spell, resolve the round
    if game.ready_to_resolve():
        game.resolve_round()

        # Broadcast updated state to all players
        socketio.emit("state_update", game.serialize(), to=room_id)

# -------------------------------------------------------
# Cloud Run Entry Point
# -------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Running on port {port}")
    socketio.run(app, host="0.0.0.0", port=port)
