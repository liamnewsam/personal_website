from Deck import *
from utils import *
from Card import *

import random

ACTIONS= ["Choose Card", "Pass", "Print State"]


class Player:
    def __init__(self, name: str, school: str, deck: Deck):
        self.name = name
        self.health = 300
        self.mana = 300
        self.school = school
        self.deck = deck
        self.boost = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.resistance = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.flat_boost = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.flat_resistance = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.accuracy = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.critical = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.block = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.armor_piercing = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.pip_conversion = {"myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.power_pip_chance = 0
        self.school_pip_chance = 0
        self.stun_resistance = 0
        self.healing_in = 0
        self.healing_out = 0

        self.active = False
        self.pips = {"regular": 0, "powerpip": 0, "myth": 0, "life": 0, "fire": 0, "ice": 0, "storm": 0, "death": 0, "balance": 0}
        self.school_pip_select = self.school
        self.shields = []
        self.traps = []
        self.charms = []
        self.auras = []
        self.dots = []
        self.hots = []

        self.struggle_counter = 1

    def act(self):
        self.active = True
        i = f"{self.name}, Choose action:"
        action = ACTIONS[int(inp(ACTIONS))-1]

        
        if action == "Choose Card": #Choosing Card
            hand_str = fancy_print_hand(self.deck.play_hand)
            #print(hand_str)
            #card_i = int(inp(self.deck.play_hand, custom=hand_str))-1
            #print(fancy_print_card(self.deck.play_hand[card_i]))
            return (action, None)
        
        if action == "Pass":
            return ("Pass", None)

        if action == "Print State":
            return("Print State", None)
    def receive_pip(self):
        if random.randint(0, 99) < self.power_pip_chance:
            print("Receiving powerpip!")
            if random.randint(0, 99) < self.school_pip_chance:
                print(f"Converting to {self.school_pip_select} pip!")
                self.pips[self.school_pip_select] += 1
            else:
                self.pips["powerpip"] += 1
        else:
            self.pips["regular"] += 1

    def deduct_pips(self, card):

        pips = card.card_def.pips
        if type(pips) == int:
            pips = {"regular": pips}
        
        for key, value in pips.items():
            if key != "regular":
                self.pips[key] -= value
        
        school_aligned = card.card_def.school == self.school
        needed_regular = pips["regular"] 
        while needed_regular > 1:
            if self.pips["regular"] > 1:
                needed_regular -= 1
                self.pips["regular"] -= 1
            elif school_aligned:
                if self.pips["powerpip"] > 0:
                    needed_regular -= 2
                    self.pips["powerpip"] -= 1
                else:
                    available = [key for key, value in self.pips.items() if key not in ["regular", "powerpip"] and value > 0]
                    random_school_pip = random.choice(available)
                    needed_regular -= 2
                    self.pips[random_school_pip] -= 1
            else:
                if self.pips["regular"] == 1:
                    needed_regular -= 1
                    self.pips["regular"] -= 1
                elif self.pips["powerpip"] > 0:
                    needed_regular -= 1
                    self.pips["powerpip"] -= 1
                else:
                    available = [key for key, value in self.pips.items() if key not in ["regular", "powerpip"] and value > 0]
                    random_school_pip = random.choice(available)
                    needed_regular -= 1
                    self.pips[random_school_pip] -= 1
        
        if needed_regular == 1:
            if self.pips["regular"] > 0:
                needed_regular -= 1
                self.pips["regular"] -= 1
            elif self.pips["powerpip"] > 0:
                needed_regular -= 1
                self.pips["powerpip"] -= 1
                if random.randint(0, 99) < self.pip_conversion[card.card_def.school]:
                    print("Converted pip!")
                    self.pips["regular"] += 1
            else:
                needed_regular -= 1
                available = [key for key, value in self.pips.items() if value > 0]
                random_school_pip = random.choice(available)
                self.pips[random_school_pip] -= 1
                if random.randint(0, 99) < self.pip_conversion[card.card_def.school]:
                    print("Converted pip!")
                    self.pips["regular"] += 1
 
    def absorb_shield(self, damage):
        """Absorb damage using the FIRST APPLICABLE SHIELD — Wizard101 rule."""
        if not self.shields:
            return 0

        # First shield only
        shield = self.shields[0]
        absorbed = shield.absorb(damage)
        return absorbed

    def refresh(self):
        self.deck.refresh()
        self.pips = {"regular": 0, "powerpip": 0}
        self.active = False

    def consume_charms(self, card, consumed_charms):
        used_charms = []
        damages = [{"school": effect.get("school", "any")} for effect in card.card_def.effects if effect["type"] in ["damage", "DoT"]]
        if damages:
            for charm in self.charms:
                if charm.type == "damage" and {charm.school} in ["any", self.school] and not isRedundant(used_charms, charm):
                    consumed_charms["damage"][charm.school] += charm.amount
                    self.charms.remove(charm)
                    used_charms.append(charm)
                    print(f"Adding {charm.amount}% damage")

    def __str__(self):
        return self.name
    

#-------------------------------- EFFECTS ------------------------------------------------
from abc import ABC, abstractmethod
import random


def isRedundant(effects, effect):
    for e in effects:
        if e.type == effect.type and e.school == effect.school and e.aspect == effect.aspect and e.amount == effect.amount:
            return True
    return False

class Effect(ABC):
    """
    Base effect class. 
    Subclasses implement resolve().
    """
    def __init__(self, template, owner, target, card):
        self.template = template
        self.owner = owner
        self.target = target
        self.card = card
        self.school = template.get("school", "any")

    @abstractmethod
    def resolve(self, game):
        """Apply the effect once (may be immediate or may add to linger list)."""
        pass

    def tick(self, game):
        """Called each round if the effect lingers."""
        if self.duration is None:
            return

        self.duration -= 1
        if self.duration <= 0:
            self.expire(game)

    def expire(self, game):
        """Optional cleanup hook."""
        pass

    # -------- Utility for subclasses --------
    def get_amount(self):
        """Handles: amount = X  or min/max random amounts."""
        if "amount" in self.template:
            return self.template["amount"]
        if "min" in self.template and "max" in self.template:
            return random.randint(self.template["min"], self.template["max"])
        raise ValueError("Effect template has no amount or min/max")

class DamageEffect(Effect):
    def resolve(self, game, consumed_charms, critical_multiplier):

        base = self.get_amount()  
        dmg = base + self.owner.flat_boost[school]
        dmg *= (1 + self.owner.boost[school] / 100.0)
        blade_mult = 1.0 + consumed_charms["damage"]["any"] / 100.0
                    
        if self.school != "any":
            blade_mult += consumed_charms["damage"][self.school] / 100.0

        trap_mult = 1.0
        used_traps = []
        for trap in self.target.traps:
            if trap.school in [self.school, "any"] and not isRedundant(used_traps, trap):
                trap_mult *= (1 + trap.amount / 100.0)
                self.target.traps.remove(trap)
                used_traps.append(trap)
                print(f"Adding {trap.amount}% damage")

        dmg *= blade_mult * trap_mult

        dmg *= critical_multiplier


        dmg -= self.target.flat_resistance[school]

        resist = self.target.resistance[school]
        pierce = self.owner.armor_piercing[school]

        effective_resist = max(resist - pierce, 0)
        dmg *= (1 - effective_resist / 100.0)


        if dmg < 0:
            dmg = 0

        

        absorbed = self.target.absorb_shield(dmg)
        dmg -= absorbed

        if dmg < 0:
            dmg = 0

        self.target.health -= dmg

        print(f"{self.owner.name} deals {int(dmg)} {school} damage to {self.target.name}!")

        return "resolved"

    

class HealEffect(Effect):
    def resolve(self, game):
        amount = self.get_amount()
        self.target.health += amount
        print(f"{self.owner.name} heals {self.target.name} for {amount}!")
        return "resolved"
    
class ShieldEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()
        self.school = template.get("school", None)

        self.is_lingering = True  # shields always linger

    def resolve(self, game):
        # Add the shield to target's shield list
        self.target.shields.append(self)
        print(f"{self.target.name} gains a {self.amount} shield!")
        return "applied"

    def absorb(self, incoming_amount):
        """Return how much of incoming damage is absorbed."""
        absorbed = min(incoming_amount, self.amount)
        self.amount -= absorbed

        if self.amount <= 0:
            # shield breaks
            if self in self.target.shields:
                self.target.shields.remove(self)

        return absorbed
    
class TrapEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

    def resolve(self, game):
        self.target.traps.append(self)
        print(f"{self.target.name} is trapped with +{self.amount} incoming damage.")
        return "applied"

    def trigger(self, base_damage):
        """Triggered when target is next hit."""
        boosted = int(base_damage * (1 + self.amount / 100))
        self.target.traps.remove(self)
        return boosted

class DoTEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class HoTEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class AuraEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class ChanceEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class CharmEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()
        self.aspect = template["aspect"]
    def resolve(self, game):
        self.target.charms.append(self)
        print(f"{self.target.name} gains a {self.amount}% {self.aspect} blade!")
        return "applied"

class DestroyEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class DetonateEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class DispelEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class EmpowerEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class ExtendEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class GambitEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class GlobalEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()
        self.aspect = template["aspect"]
        
    def resolve(self, game):
        if game.global_effect:
            print(f"Overriding global effect: {game.global_effect}")
        game.global_effect = self

class PipEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class PrismEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class RepeatEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class ReshuffleEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class StealEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()

class TakeEffect(Effect):
    def __init__(self, template, owner, target, card):
        super().__init__(template, owner, target, card)
        self.amount = self.get_amount()



EFFECT_TYPE_TO_CLASS = {
    "DoT": DoTEffect,
    "HoT": HoTEffect,
    "aura": AuraEffect,
    "chance": ChanceEffect,
    "charm": CharmEffect,
    "damage": DamageEffect,
    "destroy": DestroyEffect,
    "detonate": DetonateEffect,
    "dispel": DispelEffect,
    "empower": EmpowerEffect,
    "extend": ExtendEffect,
    "gambit": GambitEffect,
    "global": GlobalEffect,
    "heal": HealEffect,
    "pip": PipEffect,
    "prism": PrismEffect,
    "repeat": RepeatEffect,
    "reshuffle": ReshuffleEffect,
    "shield": ShieldEffect,
    "steal": StealEffect,
    "take": TakeEffect,
    "trap": TrapEffect
}

