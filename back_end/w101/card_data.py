CARD_DATA = {
    "fire_cat": {
        "name": "Fire Cat",
        "school": "Fire",
        "type": "damage",
        "description": "Deal between 80-120",
        "pip_cost": 1,
        "effects": [
            {"type": "damage", "min": 80, "max": 120, "target":"enemy_1"}
        ],
    },
    "tower_shield": {
        "name": "Tower Shield",
        "school": "Ice",
        "type": "ward",
        "description": "Applies a -50% Damage Ward",
        "pip_cost": 0,
        "effects": [
            {"type": "shield", "value": 50, "target": "self"}
        ]
    }
}