from typing import *
#--------------
from Player import *
from Card import *
from Deck import *
from utils import *

def neededTargetType(card_def):
    if card_def.condition:
        return card_def["target"]

    all_targets = set([effect["target"] for effect in card_def.effects if "target" in effect.keys()])

    if "enemy" in all_targets:
        return "enemy"
    if "ally" in all_targets:
        return "ally"
    if "enemy_selected" in all_targets:
        return "enemy_selected"
    if "ally_selected" in all_targets:
        return "ally_selected"
    
    if "self" in all_targets or "enemy_all" in all_targets or "ally_all" in all_targets:
        return None
    
    all_effect_types = set([effect["type"] for effect in card_def.effects])
    #Defaulting
    if {"damage", "trap"} & all_effect_types:
        return "enemy"
    if card_def.type == "heal":
        return "ally"

def getEffectTargetType(effect):
    if "target" in effect.keys():
        print(effect["target"])
        return effect["target"]
    
    #Defaulting
    if effect["type"] in ["damage", "trap"]:
        return "enemy"

class Game():
    def __init__(self, teamA: List[Player], teamB: List[Player]):
        self.teams = [teamA, teamB]
        self.playing_team_i = 0
        self.global_effect = None
        self.turns = 0
        self.ended = False
    
    def begin(self):
        for team in self.teams:
            for player in team:
                player.refresh()

    def play_round(self):

        for player in self.teams[self.playing_team_i]:
            self.process_ongoing_effects(player)

        for player in self.teams[self.playing_team_i]:
            self.struggle_punish(player)
            player.deck.draw_cards()
            player.receive_pip()

        casts = []
        for player in self.teams[self.playing_team_i]:
            player.active = True
            cast = None
            print(f"{player.name}, choose your action:")
            while player.active:
                action, value = player.act()
                if action == "Choose Card":
                    valid_cards = [card for card in player.deck.play_hand if self.is_playable(player, card)]
                    if len(valid_cards) == 0:
                        print("No valid cards to play!")
                        continue
                    card_i = int(inp(valid_cards, custom=fancy_print_hand(valid_cards)))-1
                    print(fancy_print_card(player.deck.play_hand[card_i]))
                    card = player.deck.play_hand[card_i]
                    cast = {"card": card, "caster": player, "actions": []}

                    targetType = neededTargetType(card.card_def) #targetType == None, or ally, enemy, ally_selected, enemy_selected

                    target = None
                    if targetType == "enemy":
                        valid_targets = [t for t in self.opposite_team() if self.conditionMet(player, card.card_def.condition, t)]
                        print("Choose an enemy:")
                        target = valid_targets[int(inp([p.name for p in valid_targets]))-1]
                    elif targetType == "ally":
                        valid_targets = [t for t in self.current_team() if self.conditionMet(player, card.card_def.condition, t)]
                        print("Choose an ally: ")
                        target = valid_targets[int(inp([p.name for p in valid_targets]))-1]

                    for effect in card.card_def.effects:
                        targetType = getEffectTargetType(effect)
                        
                        if targetType == "self":
                            cast["actions"].append({"effect": effect, "target": player})
                        elif targetType in ["enemy", "enemy_same", "enemy_selected", "ally", "ally_same", "ally_selected"]:
                            cast["actions"].append({"effect": effect, "target": target})
                        elif targetType == "enemy_all":
                            cast["actions"].extend([{"effect": effect, "target": t} for t in self.opposite_team()])
                        elif targetType == "ally_all":
                            cast["actions"].extend([{"effect": effect, "target": t} for t in self.current_team()])
                        
                    casts.append(cast)
                    player.active = False
                elif action == "Pass":
                    if cast:
                        casts.append(cast)
                    player.active = False
                
                elif action == "Print State":
                    print(self)
        

        for cast in casts:

            consumed_charms = {
                "dispelled": False, 
                "damage": {"any": 0, "myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0},
                "armor_piercing": {"any": 0, "myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0},
                "health": {"in": 0, "out": 0},
                "accuracy": 0
            }

            player.consume_charms(cast["card"], consumed_charms)

            if consumed_charms["dispelled"]:
                print("Dispelled!")

            if not consumed_charms["dispelled"] and random.randint(0, 99) < cast["card"].card_def.accuracy + consumed_charms["accuracy"]:
                print(f"{cast['caster']} casts {cast['card']}!")
                cast["caster"].deduct_pips(cast["card"])

                critical_multiplier = 1
                if cast["card"].hasEffect(["damage", "DoT", "heal", "HoT"]):
                    if random.randint(0, 99) < cast["caster"].critical[cast["card"].card_def.school]:         # returns True/False
                        if not cast["card"].hasEffect(["damage", "DoT"]) or not random.randint(0, 99) < cast["target"].block[cast["card"].card_def.school]:   # returns True/False
                            critical_multiplier = 2                        # Wizard101 crit multiplier
                            print(f"🔥 CRITICAL HIT by {self.owner.name}!")

                print("Effects:")
                for action in cast["actions"]:
                    print("HI")
                    self.resolve_action(cast["card"], cast["caster"], action["target"], action["effect"], consumed_charms, critical_multiplier)


            else:
                print("Fizzled!")

            cast['caster'].deck.play_hand.remove(cast["card"])
            cast["caster"].deck.play_discard.append(cast["card"])

        self.playing_team_i = (self.playing_team_i+1) % len(self.teams)
        self.turns += 1

    def is_playable(self, player, card):
        test = player.pips.copy()
        player_school = player.school
        card_school = card.card_def.school
        required_pips = card.card_def.pips
        school_aligned = player_school == card_school
        if type(required_pips) == dict:
            for school in [x for x in required_pips.keys() if x != "regular"]:
                test[school] -= required_pips[school]
        else:
            required_pips = {"regular": card.card_def.pips}    
        total_regular = 0
        for key, value in test.items():
            if value < 0:
                return False
            
            total_regular += value*(2 if school_aligned and key != "regular" else 1)
        
        if total_regular < required_pips["regular"]:
            return False

        if card.card_def.condition == None:
            return True

    def interpretTargetType(self, player, type):
        if type == "enemy":
            return self.opposite_team()
        if type == "ally":
            return self.current_team()
        if type == "self":
            return [player]

    def opposite_team(self):
        return self.teams[(self.playing_team_i+1) % len(self.teams)]
    
    def current_team(self):
        return self.teams[self.playing_team_i]

    def conditionMet(self, player, condition, target):
        if not condition:
            return True
        if condition["type"] == "amount":
            if target.hasAspectAmount(condition["aspect"], condition["amount"]):
                return True
        
        return False
    
    def resolve_action(self, card: Card, caster: Player, target: Player, effect_template: dict, consumed_charms, critical_multiplier):

        effect = EFFECT_TYPE_TO_CLASS[effect_template["type"]](effect_template, caster, target, card)
        if type(effect) in [DamageEffect, HealEffect, DoTEffect, HoTEffect]:
            result = effect.resolve(self, consumed_charms, critical_multiplier)
        else:
            result = effect.resolve(self)
        print(result)
        return result

    def struggle_punish(self, player):
        if len(player.deck.play_cards) == 0 and len(self.playable_cards(player)) == 0:
            player.health -= 10 * player.struggle_counter
            player.struggle_counter += 1

    def process_ongoing_effects(self, player):

        #Process DoTs
        remaining = []
        for dot in player.dots:
            alive = dot.tick(self)
            if alive:
                remaining.append(dot)
            else:
                print(f"{dot.type} on {player.name} expired.")
        player.dots = remaining

        #Process HoTs
        remaining = []
        for hot in player.hots:
            alive = hot.tick(self)
            if alive:
                remaining.append(hot)
            else:
                print(f"{hot.type} on {player.name} expired.")
        player.hot = remaining

    def __str__(self):
        s = ''
        for team_name, team in zip(["Team A", "Team B"], self.teams):
            s += team_name + ":\n"
            for player in team:
                s += f"\t{player.name}:\n"
                s += f"\t\tHealth: {player.health}\n"
                s += f"\t\tMana: {player.mana}\n"
                s += f"\t\tSchool: {player.school}\n"
                s += f"\t\tPips: {player.pips}\n"
                s += f"\t\tShields: {player.shields}\n"
                s += f"\t\tCharms: {player.charms}\n"
                s += f"\t\tTraps: {player.traps}\n"
                s += f"\t\tDoTs: {player.dots}\n"
                s += f"\t\tHoTs: {player.hots}\n"
        
        return s