"""Ultra-advanced Clash Royale AI with human-like tactical gameplay and enemy tracking."""

import collections
import random
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Protocol, Tuple, List, Dict, Set
import logging

from pyclashbot.bot.card_detection import (
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_play_coords_for_card,
    switch_side,
)
from pyclashbot.bot.nav import (
    check_for_in_battle_with_delay,
    check_for_trophy_reward_menu,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    get_to_activity_log,
    handle_trophy_reward_menu,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.bot.recorder import save_image, save_play, save_win_loss
from pyclashbot.detection.image_rec import (
    check_line_for_color,
    find_image,
    pixel_is_equal,
)
from pyclashbot.utils.logger import Logger


class BattlePhase(Enum):
    OPENING = "opening"
    EARLY = "early"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    OVERTIME = "overtime"


class CardType(Enum):
    TANK = "tank"
    DPS = "dps"
    SWARM = "swarm"
    SPELL = "spell"
    BUILDING = "building"
    WIN_CONDITION = "win_condition"
    SUPPORT = "support"
    ANTI_AIR = "anti_air"


class Lane(Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class PlayStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    COUNTER_PUSH = "counter_push"
    CYCLE = "cycle"
    BEATDOWN = "beatdown"
    CONTROL = "control"
    SIEGE = "siege"


@dataclass
class EnemyUnit:
    position: Tuple[int, int]
    unit_type: CardType
    timestamp: float
    threat_level: ThreatLevel
    lane: Lane
    health_estimate: float = 1.0
    moving_towards: Optional[Tuple[int, int]] = None
    predicted_path: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class GameState:
    our_elixir: int = 0
    enemy_elixir_estimate: int = 0
    our_tower_health: Dict[str, float] = field(default_factory=lambda: {"king": 1.0, "left": 1.0, "right": 1.0})
    enemy_tower_health: Dict[str, float] = field(default_factory=lambda: {"king": 1.0, "left": 1.0, "right": 1.0})
    enemy_units: List[EnemyUnit] = field(default_factory=list)
    last_enemy_plays: collections.deque = field(default_factory=lambda: collections.deque(maxlen=8))
    enemy_deck_revealed: Set[str] = field(default_factory=set)
    pressure_lanes: Dict[Lane, ThreatLevel] = field(default_factory=lambda: {
        Lane.LEFT: ThreatLevel.NONE,
        Lane.RIGHT: ThreatLevel.NONE,
        Lane.CENTER: ThreatLevel.NONE
    })


@dataclass(frozen=True)
class BattlefieldZones:
    ENEMY_SPAWN_LEFT: Tuple[int, int] = (150, 50)
    ENEMY_SPAWN_RIGHT: Tuple[int, int] = (490, 50)
    ENEMY_SPAWN_CENTER: Tuple[int, int] = (320, 80)
    
    OUR_SPAWN_LEFT: Tuple[int, int] = (150, 520)
    OUR_SPAWN_RIGHT: Tuple[int, int] = (490, 520)
    OUR_SPAWN_CENTER: Tuple[int, int] = (320, 490)
    
    BRIDGE_LEFT: Tuple[int, int] = (150, 285)
    BRIDGE_RIGHT: Tuple[int, int] = (490, 285)
    BRIDGE_CENTER: Tuple[int, int] = (320, 285)
    
    KING_TOWER_ENEMY: Tuple[int, int] = (320, 120)
    KING_TOWER_OURS: Tuple[int, int] = (320, 450)
    
    PRINCESS_TOWER_ENEMY_LEFT: Tuple[int, int] = (200, 160)
    PRINCESS_TOWER_ENEMY_RIGHT: Tuple[int, int] = (440, 160)
    PRINCESS_TOWER_OURS_LEFT: Tuple[int, int] = (200, 410)
    PRINCESS_TOWER_OURS_RIGHT: Tuple[int, int] = (440, 410)
    
    POCKET_LEFT: Tuple[int, int] = (80, 285)
    POCKET_RIGHT: Tuple[int, int] = (560, 285)
    
    ANTI_PUSH_LEFT: Tuple[int, int] = (200, 350)
    ANTI_PUSH_RIGHT: Tuple[int, int] = (440, 350)
    
    KITING_ZONES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 400), (540, 400), (320, 380), (260, 420), (380, 420)
    ])


class EnemyTracker:
    def __init__(self):
        self.game_state = GameState()
        self.zones = BattlefieldZones()
        self.enemy_play_patterns = collections.defaultdict(int)
        self.enemy_response_patterns = {}
        self.last_scan_time = 0.0
        
    def scan_battlefield(self, emulator) -> None:
        current_time = time.time()
        if current_time - self.last_scan_time < 0.5:
            return
            
        iar = emulator.screenshot()
        self._detect_enemy_units(iar)
        self._estimate_enemy_elixir()
        self._analyze_tower_health(iar)
        self._update_threat_levels()
        self.last_scan_time = current_time
        
    def _detect_enemy_units(self, iar) -> None:
        detection_zones = [
            (self.zones.ENEMY_SPAWN_LEFT, Lane.LEFT),
            (self.zones.ENEMY_SPAWN_RIGHT, Lane.RIGHT),
            (self.zones.ENEMY_SPAWN_CENTER, Lane.CENTER),
            (self.zones.BRIDGE_LEFT, Lane.LEFT),
            (self.zones.BRIDGE_RIGHT, Lane.RIGHT),
        ]
        
        current_units = []
        current_time = time.time()
        
        for zone_pos, lane in detection_zones:
            if self._detect_unit_at_position(iar, zone_pos):
                unit = EnemyUnit(
                    position=zone_pos,
                    unit_type=self._classify_unit_type(iar, zone_pos),
                    timestamp=current_time,
                    threat_level=self._assess_threat_level(zone_pos, lane),
                    lane=lane
                )
                current_units.append(unit)
                
        self.game_state.enemy_units = current_units
        
    def _detect_unit_at_position(self, iar, pos: Tuple[int, int]) -> bool:
        x, y = pos
        search_radius = 30
        
        for dx in range(-search_radius, search_radius, 5):
            for dy in range(-search_radius, search_radius, 5):
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < len(iar[0]) and 0 <= check_y < len(iar):
                    pixel = iar[check_y][check_x]
                    if self._is_enemy_unit_pixel(pixel):
                        return True
        return False
        
    def _is_enemy_unit_pixel(self, pixel) -> bool:
        enemy_colors = [
            [255, 0, 0], [200, 50, 50], [180, 30, 30],
            [255, 100, 100], [220, 80, 80]
        ]
        
        for color in enemy_colors:
            if pixel_is_equal(pixel, color, tol=40):
                return True
        return False
        
    def _classify_unit_type(self, iar, pos: Tuple[int, int]) -> CardType:
        x, y = pos
        area_pixels = []
        
        for dx in range(-15, 16, 3):
            for dy in range(-15, 16, 3):
                if 0 <= x + dx < len(iar[0]) and 0 <= y + dy < len(iar):
                    area_pixels.append(iar[y + dy][x + dx])
        
        avg_brightness = sum(sum(pixel) for pixel in area_pixels) / (len(area_pixels) * 3)
        
        if avg_brightness > 200:
            return CardType.SPELL
        elif avg_brightness > 150:
            return CardType.SWARM
        elif avg_brightness > 100:
            return CardType.DPS
        else:
            return CardType.TANK
            
    def _assess_threat_level(self, pos: Tuple[int, int], lane: Lane) -> ThreatLevel:
        x, y = pos
        
        if y > 400:
            return ThreatLevel.CRITICAL
        elif y > 300:
            return ThreatLevel.HIGH
        elif y > 200:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
            
    def _estimate_enemy_elixir(self) -> None:
        current_time = time.time()
        recent_plays = [play for play in self.game_state.last_enemy_plays 
                      if current_time - play.get('time', 0) < 10]
        
        estimated_spent = sum(play.get('cost', 3) for play in recent_plays)
        time_passed = max(1, current_time - min(play.get('time', current_time) for play in recent_plays)) if recent_plays else 10
        elixir_generated = time_passed * 1.4
        
        self.game_state.enemy_elixir_estimate = max(0, min(10, int(elixir_generated - estimated_spent)))
        
    def _analyze_tower_health(self, iar) -> None:
        tower_positions = {
            "enemy_left": self.zones.PRINCESS_TOWER_ENEMY_LEFT,
            "enemy_right": self.zones.PRINCESS_TOWER_ENEMY_RIGHT,
            "enemy_king": self.zones.KING_TOWER_ENEMY,
            "our_left": self.zones.PRINCESS_TOWER_OURS_LEFT,
            "our_right": self.zones.PRINCESS_TOWER_OURS_RIGHT,
            "our_king": self.zones.KING_TOWER_OURS,
        }
        
        for tower, pos in tower_positions.items():
            health = self._estimate_tower_health(iar, pos)
            if "enemy" in tower:
                tower_key = tower.replace("enemy_", "")
                self.game_state.enemy_tower_health[tower_key] = health
            else:
                tower_key = tower.replace("our_", "")
                self.game_state.our_tower_health[tower_key] = health
                
    def _estimate_tower_health(self, iar, pos: Tuple[int, int]) -> float:
        x, y = pos
        health_bar_y = y - 20
        
        total_green = 0
        total_pixels = 0
        
        for check_x in range(x - 25, x + 26):
            if 0 <= check_x < len(iar[0]) and 0 <= health_bar_y < len(iar):
                pixel = iar[health_bar_y][check_x]
                total_pixels += 1
                
                if pixel_is_equal(pixel, [0, 255, 0], tol=50):
                    total_green += 1
                    
        return total_green / total_pixels if total_pixels > 0 else 1.0
        
    def _update_threat_levels(self) -> None:
        for lane in Lane:
            lane_units = [unit for unit in self.game_state.enemy_units if unit.lane == lane]
            if not lane_units:
                self.game_state.pressure_lanes[lane] = ThreatLevel.NONE
            else:
                max_threat = max(unit.threat_level for unit in lane_units)
                self.game_state.pressure_lanes[lane] = max_threat
                
    def get_most_threatened_lane(self) -> Lane:
        return max(self.game_state.pressure_lanes.keys(), 
                  key=lambda lane: self.game_state.pressure_lanes[lane].value)
                  
    def predict_enemy_next_move(self) -> Dict[str, any]:
        if len(self.game_state.last_enemy_plays) < 3:
            return {"prediction": "unknown", "confidence": 0.0}
            
        recent_patterns = list(self.game_state.last_enemy_plays)[-3:]
        pattern_hash = tuple(play.get('lane', 'center') for play in recent_patterns)
        
        if pattern_hash in self.enemy_response_patterns:
            prediction = self.enemy_response_patterns[pattern_hash]
            return {"prediction": prediction, "confidence": 0.8}
            
        return {"prediction": "defensive", "confidence": 0.3}


class TacticalAI:
    def __init__(self, enemy_tracker: EnemyTracker):
        self.enemy_tracker = enemy_tracker
        self.zones = BattlefieldZones()
        self.current_strategy = PlayStyle.DEFENSIVE
        self.cycle_count = 0
        self.last_push_time = 0.0
        self.elixir_advantage_threshold = 2
        
    def analyze_situation(self) -> Dict[str, any]:
        game_state = self.enemy_tracker.game_state
        
        analysis = {
            "elixir_advantage": game_state.our_elixir - game_state.enemy_elixir_estimate,
            "tower_advantage": self._calculate_tower_advantage(),
            "pressure_situation": self._analyze_pressure(),
            "recommended_play_style": self._determine_play_style(),
            "priority_lane": self._select_priority_lane(),
            "defensive_needs": self._assess_defensive_needs(),
            "push_opportunity": self._evaluate_push_opportunity()
        }
        
        return analysis
        
    def _calculate_tower_advantage(self) -> float:
        our_towers = sum(self.enemy_tracker.game_state.our_tower_health.values())
        enemy_towers = sum(self.enemy_tracker.game_state.enemy_tower_health.values())
        return our_towers - enemy_towers
        
    def _analyze_pressure(self) -> Dict[str, any]:
        pressure_lanes = self.enemy_tracker.game_state.pressure_lanes
        total_pressure = sum(threat.value for threat in pressure_lanes.values())
        
        return {
            "total_pressure": total_pressure,
            "max_pressure_lane": max(pressure_lanes.keys(), key=lambda l: pressure_lanes[l].value),
            "needs_immediate_defense": total_pressure >= ThreatLevel.HIGH.value
        }
        
    def _determine_play_style(self) -> PlayStyle:
        analysis = self._analyze_pressure()
        elixir_diff = self.enemy_tracker.game_state.our_elixir - self.enemy_tracker.game_state.enemy_elixir_estimate
        
        if analysis["needs_immediate_defense"]:
            return PlayStyle.DEFENSIVE
        elif elixir_diff >= 3:
            return PlayStyle.AGGRESSIVE
        elif elixir_diff <= -2:
            return PlayStyle.CYCLE
        else:
            return PlayStyle.COUNTER_PUSH
            
    def _select_priority_lane(self) -> Lane:
        pressure_lanes = self.enemy_tracker.game_state.pressure_lanes
        tower_health = self.enemy_tracker.game_state.enemy_tower_health
        
        if pressure_lanes[Lane.LEFT] >= ThreatLevel.HIGH:
            return Lane.LEFT
        elif pressure_lanes[Lane.RIGHT] >= ThreatLevel.HIGH:
            return Lane.RIGHT
            
        if tower_health.get("left", 1.0) < tower_health.get("right", 1.0):
            return Lane.LEFT
        elif tower_health.get("right", 1.0) < tower_health.get("left", 1.0):
            return Lane.RIGHT
        else:
            return Lane.LEFT if random.random() < 0.5 else Lane.RIGHT
            
    def _assess_defensive_needs(self) -> List[Dict[str, any]]:
        needs = []
        
        for unit in self.enemy_tracker.game_state.enemy_units:
            if unit.threat_level >= ThreatLevel.HIGH:
                counter_pos = self._calculate_counter_position(unit)
                needs.append({
                    "threat": unit,
                    "counter_position": counter_pos,
                    "recommended_counter": self._recommend_counter_card(unit),
                    "urgency": unit.threat_level.value
                })
                
        return sorted(needs, key=lambda x: x["urgency"], reverse=True)
        
    def _calculate_counter_position(self, threat: EnemyUnit) -> Tuple[int, int]:
        x, y = threat.position
        
        if threat.unit_type == CardType.TANK:
            return (x, min(y + 80, 450))
        elif threat.unit_type == CardType.SWARM:
            return (x + random.randint(-40, 40), y + 60)
        else:
            return (x + random.randint(-30, 30), y + 50)
            
    def _recommend_counter_card(self, threat: EnemyUnit) -> str:
        counters = {
            CardType.TANK: ["building", "swarm", "dps"],
            CardType.SWARM: ["spell", "splash"],
            CardType.DPS: ["tank", "building"],
            CardType.SPELL: ["none"],
            CardType.BUILDING: ["spell", "tank"]
        }
        
        return random.choice(counters.get(threat.unit_type, ["generic"]))
        
    def _evaluate_push_opportunity(self) -> Dict[str, any]:
        current_time = time.time()
        time_since_last_push = current_time - self.last_push_time
        
        elixir_advantage = (self.enemy_tracker.game_state.our_elixir - 
                           self.enemy_tracker.game_state.enemy_elixir_estimate)
        
        no_immediate_threats = all(
            threat.value < ThreatLevel.HIGH.value 
            for threat in self.enemy_tracker.game_state.pressure_lanes.values()
        )
        
        opportunity_score = 0
        if elixir_advantage >= 2:
            opportunity_score += 3
        if no_immediate_threats:
            opportunity_score += 2
        if time_since_last_push > 15:
            opportunity_score += 1
            
        return {
            "should_push": opportunity_score >= 4,
            "opportunity_score": opportunity_score,
            "recommended_lane": self._select_priority_lane(),
            "push_type": "beatdown" if elixir_advantage >= 4 else "pressure"
        }


class AdvancedPlayAreaGenerator:
    def __init__(self):
        self.zones = BattlefieldZones()
        self.tactical_ai = None
        
    def set_tactical_ai(self, tactical_ai: TacticalAI):
        self.tactical_ai = tactical_ai
        
    def get_optimal_play_position(
        self, 
        card_type: CardType, 
        target_lane: Lane,
        play_intent: str = "offensive"
    ) -> Tuple[int, int]:
        
        if play_intent == "defensive" and self.tactical_ai:
            return self._get_defensive_position(card_type, target_lane)
        elif play_intent == "counter":
            return self._get_counter_position(card_type, target_lane)
        elif play_intent == "push":
            return self._get_push_position(card_type, target_lane)
        else:
            return self._get_standard_position(card_type, target_lane)
            
    def _get_defensive_position(self, card_type: CardType, lane: Lane) -> Tuple[int, int]:
        if card_type == CardType.BUILDING:
            if lane == Lane.LEFT:
                return (220, 380)
            elif lane == Lane.RIGHT:
                return (420, 380)
            else:
                return (320, 390)
                
        elif card_type in [CardType.TANK, CardType.DPS]:
            base_positions = {
                Lane.LEFT: self.zones.ANTI_PUSH_LEFT,
                Lane.RIGHT: self.zones.ANTI_PUSH_RIGHT,
                Lane.CENTER: (320, 360)
            }
            
            base_x, base_y = base_positions[lane]
            return (base_x + random.randint(-25, 25), base_y + random.randint(-15, 15))
            
        else:
            return self._get_kiting_position(lane)
            
    def _get_counter_position(self, card_type: CardType, lane: Lane) -> Tuple[int, int]:
        if self.tactical_ai:
            threats = self.tactical_ai.enemy_tracker.game_state.enemy_units
            lane_threats = [t for t in threats if t.lane == lane and t.threat_level >= ThreatLevel.MEDIUM]
            
            if lane_threats:
                threat = max(lane_threats, key=lambda t: t.threat_level.value)
                return self.tactical_ai._calculate_counter_position(threat)
                
        return self._get_defensive_position(card_type, lane)
        
    def _get_push_position(self, card_type: CardType, lane: Lane) -> Tuple[int, int]:
        if card_type == CardType.TANK:
            bridge_positions = {
                Lane.LEFT: self.zones.BRIDGE_LEFT,
                Lane.RIGHT: self.zones.BRIDGE_RIGHT,
                Lane.CENTER: self.zones.BRIDGE_CENTER
            }
            
            base_x, base_y = bridge_positions[lane]
            return (base_x + random.randint(-20, 20), base_y + random.randint(-10, 10))
            
        elif card_type == CardType.WIN_CONDITION:
            pocket_positions = {
                Lane.LEFT: self.zones.POCKET_LEFT,
                Lane.RIGHT: self.zones.POCKET_RIGHT,
                Lane.CENTER: (320, 300)
            }
            
            return pocket_positions[lane]
            
        else:
            return self._get_support_position(lane)
            
    def _get_support_position(self, lane: Lane) -> Tuple[int, int]:
        support_positions = {
            Lane.LEFT: (180, 320),
            Lane.RIGHT: (460, 320),
            Lane.CENTER: (320, 330)
        }
        
        base_x, base_y = support_positions[lane]
        return (base_x + random.randint(-30, 30), base_y + random.randint(-20, 20))
        
    def _get_kiting_position(self, lane: Lane) -> Tuple[int, int]:
        lane_kiting_spots = {
            Lane.LEFT: [(100, 400), (160, 420), (120, 380)],
            Lane.RIGHT: [(540, 400), (480, 420), (520, 380)],
            Lane.CENTER: [(320, 380), (280, 400), (360, 400)]
        }
        
        return random.choice(lane_kiting_spots[lane])
        
    def _get_standard_position(self, card_type: CardType, lane: Lane) -> Tuple[int, int]:
        standard_positions = {
            Lane.LEFT: (150, 350),
            Lane.RIGHT: (490, 350),
            Lane.CENTER: (320, 340)
        }
        
        base_x, base_y = standard_positions[lane]
        return (base_x + random.randint(-40, 40), base_y + random.randint(-30, 30))
        
    def get_spell_position(
        self, 
        target_area: Tuple[int, int], 
        spell_radius: int = 60,
        predict_movement: bool = True
    ) -> Tuple[int, int]:
        
        base_x, base_y = target_area
        
        if predict_movement and self.tactical_ai:
            enemy_units = self.tactical_ai.enemy_tracker.game_state.enemy_units
            nearby_units = [
                unit for unit in enemy_units 
                if abs(unit.position[0] - base_x) < spell_radius and 
                   abs(unit.position[1] - base_y) < spell_radius
            ]
            
            if nearby_units:
                avg_x = sum(unit.position[0] for unit in nearby_units) / len(nearby_units)
                avg_y = sum(unit.position[1] for unit in nearby_units) / len(nearby_units)
                
                prediction_offset = 15
                predicted_y = avg_y + prediction_offset
                
                return (int(avg_x), int(predicted_y))
                
        return (base_x + random.randint(-20, 20), base_y + random.randint(-15, 15))


class EliteCardSelector:
    def __init__(self):
        self.recent_cards = collections.deque(maxlen=4)
        self.card_synergies = {
            0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]
        }
        self.situation_priorities = {
            "defense": [CardType.BUILDING, CardType.SWARM, CardType.SPELL],
            "push": [CardType.WIN_CONDITION, CardType.TANK, CardType.SUPPORT],
            "counter": [CardType.DPS, CardType.SPELL, CardType.SWARM],
            "cycle": [CardType.SUPPORT, CardType.SWARM]
        }
        
    def select_optimal_card(
        self, 
        available_indices: List[int],
        situation: str = "neutral",
        tactical_ai: TacticalAI = None
    ) -> Tuple[int, CardType, str]:
        
        if not available_indices:
            raise ValueError("No cards available")
            
        if tactical_ai:
            analysis = tactical_ai.analyze_situation()
            situation = self._determine_situation_from_analysis(analysis)
            
        scored_cards = []
        
        for card_index in available_indices:
            card_type = self._estimate_card_type(card_index)
            score = self._calculate_card_score(
                card_index, card_type, situation, tactical_ai
            )
            
            play_intent = self._determine_play_intent(card_type, situation)
            
            scored_cards.append((card_index, card_type, play_intent, score))
            
        best_card = max(scored_cards, key=lambda x: x[3])
        selected_index = best_card[0]
        
        if selected_index not in self.recent_cards:
            self.recent_cards.append(selected_index)
            
        return selected_index, best_card[1], best_card[2]
        
    def _determine_situation_from_analysis(self, analysis: Dict[str, any]) -> str:
        if analysis["defensive_needs"]:
            return "defense"
        elif analysis["push_opportunity"]["should_push"]:
            return "push"
        elif analysis["elixir_advantage"] <= -2:
            return "cycle"
        else:
            return "counter"
            
    def _estimate_card_type(self, card_index: int) -> CardType:
        type_mapping = {
            0: CardType.TANK,
            1: CardType.DPS,
            2: CardType.SWARM,
            3: CardType.SPELL
        }
        return type_mapping.get(card_index, CardType.SUPPORT)
        
    def _calculate_card_score(
        self, 
        card_index: int, 
        card_type: CardType, 
        situation: str,
        tactical_ai: TacticalAI = None
    ) -> float:
        
        base_score = 50.0
        
        if card_type in self.situation_priorities.get(situation, []):
            base_score += 30.0
            
        if card_index in self.recent_cards:
            recency_penalty = (4 - list(self.recent_cards).index(card_index)) * 10
            base_score -= recency_penalty
            
        if tactical_ai:
            synergy_bonus = self._calculate_synergy_bonus(card_index, tactical_ai)
            base_score += synergy_bonus
            
        return base_score + random.uniform(-5, 5)
        
    def _calculate_synergy_bonus(self, card_index: int, tactical_ai: TacticalAI) -> float:
        bonus = 0.0
        
        synergy_cards = self.card_synergies.get(card_index, [])
        recent_plays = list(self.recent_cards)[-2:]
        
        for synergy_card in synergy_cards:
            if synergy_card in recent_plays:
                bonus += 15.0
                
        return bonus
        
    def _determine_play_intent(self, card_type: CardType, situation: str) -> str:
        intent_mapping = {
            "defense": "defensive",
            "push": "push",
            "counter": "counter",
            "cycle": "cycle"
        }
        
        return intent_mapping.get(situation, "offensive")


class MasterBattleEngine:
    def __init__(self):
        self.enemy_tracker = EnemyTracker()
        self.tactical_ai = TacticalAI(self.enemy_tracker)
        self.play_area = AdvancedPlayAreaGenerator()
        self.card_selector = EliteCardSelector()
        self.zones = BattlefieldZones()
        
        self.play_area.set_tactical_ai(self.tactical_ai)
        
        self.reaction_times = {
            ThreatLevel.CRITICAL: 0.2,
            ThreatLevel.HIGH: 0.5,
            ThreatLevel.MEDIUM: 1.0,
            ThreatLevel.LOW: 2.0,
            ThreatLevel.NONE: 3.0
        }
        
        self.micro_techniques = {
            "pig_push": self._execute_pig_push,
            "spell_prediction": self._predict_and_dodge_spell,
            "kiting": self._execute_kiting,
            "split_push": self._execute_split_push,
            "spell_cycling": self._cycle_spells_optimally,
            "elixir_pump_timing": self._time_elixir_pump
        }
        
        self.macro_strategies = {
            "beatdown": self._execute_beatdown,
            "siege": self._execute_siege,
            "control": self._execute_control,
            "cycle": self._execute_cycle,
            "bridge_spam": self._execute_bridge_spam
        }
        
        self.psychological_warfare = {
            "bm_timing": self._time_bm_perfectly,
            "fake_rotation": self._fake_card_rotation,
            "tempo_manipulation": self._manipulate_game_tempo,
            "pressure_points": self._apply_psychological_pressure
        }
        
        self.battle_memory = {
            "successful_plays": collections.deque(maxlen=20),
            "failed_plays": collections.deque(maxlen=10),
            "opponent_weaknesses": {},
            "timing_patterns": collections.defaultdict(list)
        }
        
    def execute_god_tier_battle(self, emulator, logger: Logger, recording_flag: bool = False) -> bool:
        logger.change_status("Initiating God-Tier Battle AI")
        
        battle_start_time = time.time()
        self._initialize_battle_state()
        
        try:
            while check_for_in_battle_with_delay(emulator):
                current_time = time.time()
                battle_elapsed = current_time - battle_start_time
                
                self.enemy_tracker.scan_battlefield(emulator)
                analysis = self.tactical_ai.analyze_situation()
                
                decision = self._make_strategic_decision(analysis, battle_elapsed)
                
                if decision["action"] == "immediate_defense":
                    success = self._execute_emergency_defense(emulator, logger, decision)
                elif decision["action"] == "counter_attack":
                    success = self._execute_perfect_counter(emulator, logger, decision)
                elif decision["action"] == "push":
                    success = self._execute_calculated_push(emulator, logger, decision)
                elif decision["action"] == "cycle":
                    success = self._execute_smart_cycle(emulator, logger, decision)
                elif decision["action"] == "spell_prediction":
                    success = self._execute_spell_prediction(emulator, logger, decision)
                elif decision["action"] == "micro_technique":
                    success = self._execute_micro_play(emulator, logger, decision)
                else:
                    success = self._execute_standard_play(emulator, logger, decision)
                
                if success:
                    self._learn_from_play(decision, True)
                    self._apply_psychological_pressure(emulator, logger, decision)
                else:
                    self._learn_from_play(decision, False)
                
                reaction_delay = self._calculate_human_like_delay(analysis)
                time.sleep(reaction_delay)
                
                if recording_flag:
                    save_image(emulator.screenshot())
                    
        except Exception as e:
            logger.change_status(f"God-tier AI encountered error: {str(e)}")
            return False
            
        logger.change_status("God-Tier Battle completed successfully")
        return True
        
    def _initialize_battle_state(self) -> None:
        self.tactical_ai.current_strategy = PlayStyle.DEFENSIVE
        self.battle_memory["successful_plays"].clear()
        self.battle_memory["failed_plays"].clear()
        
    def _make_strategic_decision(self, analysis: Dict[str, any], battle_time: float) -> Dict[str, any]:
        base_decision = {
            "action": "standard",
            "priority": 1,
            "confidence": 0.5,
            "reasoning": "default"
        }
        
        if analysis["defensive_needs"]:
            urgent_threats = [need for need in analysis["defensive_needs"] if need["urgency"] >= 3]
            if urgent_threats:
                return {
                    "action": "immediate_defense",
                    "priority": 5,
                    "confidence": 0.95,
                    "target": urgent_threats[0],
                    "reasoning": "critical_threat_detected"
                }
        
        if analysis["push_opportunity"]["should_push"]:
            enemy_elixir = self.enemy_tracker.game_state.enemy_elixir_estimate
            our_elixir = self.enemy_tracker.game_state.our_elixir
            
            if our_elixir >= 7 and enemy_elixir <= 3:
                return {
                    "action": "push",
                    "priority": 4,
                    "confidence": 0.9,
                    "lane": analysis["priority_lane"],
                    "type": "punishment_push",
                    "reasoning": "massive_elixir_advantage"
                }
        
        if self._detect_spell_pattern(battle_time):
            return {
                "action": "spell_prediction",
                "priority": 3,
                "confidence": 0.8,
                "predicted_spell": self._predict_incoming_spell(),
                "reasoning": "spell_pattern_detected"
            }
        
        advanced_opportunity = self._detect_advanced_opportunity(analysis, battle_time)
        if advanced_opportunity:
            return advanced_opportunity
            
        return self._make_tempo_based_decision(analysis, battle_time)
        
    def _execute_emergency_defense(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        threat = decision["target"]["threat"]
        counter_pos = decision["target"]["counter_position"]
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if not available_cards:
            return False
            
        best_counter_index = self._find_best_counter_card(available_cards, threat)
        
        logger.change_status(f"EMERGENCY DEFENSE: {threat.unit_type.value} at {threat.position}")
        
        precise_counter_pos = self._calculate_precise_counter_position(threat, counter_pos)
        
        hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][best_counter_index]
        emulator.click(*hand_coord)
        time.sleep(0.05)
        emulator.click(*precise_counter_pos)
        
        self._execute_follow_up_defense(emulator, logger, threat)
        
        return True
        
    def _execute_perfect_counter(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        logger.change_status("EXECUTING PERFECT COUNTER-ATTACK")
        
        enemy_commitment = self._calculate_enemy_commitment()
        counter_lane = self._select_optimal_counter_lane(enemy_commitment)
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if len(available_cards) < 2:
            return False
            
        combo_cards = self._select_counter_combo(available_cards, counter_lane)
        
        for i, (card_index, delay, position) in enumerate(combo_cards):
            if i > 0:
                time.sleep(delay)
                
            hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][card_index]
            emulator.click(*hand_coord)
            time.sleep(0.1)
            emulator.click(*position)
            
            logger.change_status(f"Counter card {i+1}: {card_index} at {position}")
            
        self._execute_counter_support(emulator, logger, counter_lane)
        
        return True
        
    def _execute_calculated_push(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        push_type = decision.get("type", "standard")
        target_lane = decision.get("lane", Lane.LEFT)
        
        logger.change_status(f"CALCULATED PUSH: {push_type} on {target_lane.value}")
        
        if push_type == "beatdown":
            return self._execute_beatdown_push(emulator, logger, target_lane)
        elif push_type == "punishment_push":
            return self._execute_punishment_push(emulator, logger, target_lane)
        elif push_type == "split_push":
            return self._execute_split_push_advanced(emulator, logger)
        else:
            return self._execute_pressure_push(emulator, logger, target_lane)
            
    def _execute_smart_cycle(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        logger.change_status("SMART CYCLING")
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if not available_cards:
            return False
            
        cycle_card = self._select_optimal_cycle_card(available_cards)
        cycle_position = self._calculate_cycle_position()
        
        hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][cycle_card]
        emulator.click(*hand_coord)
        time.sleep(0.1)
        emulator.click(*cycle_position)
        
        return True
        
    def _execute_spell_prediction(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        predicted_spell = decision["predicted_spell"]
        logger.change_status(f"SPELL PREDICTION: Dodging {predicted_spell}")
        
        dodge_positions = self._calculate_spell_dodge_positions(predicted_spell)
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if not available_cards:
            return False
            
        bait_card = available_cards[0]
        dodge_pos = random.choice(dodge_positions)
        
        hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][bait_card]
        emulator.click(*hand_coord)
        time.sleep(0.1)
        emulator.click(*dodge_pos)
        
        time.sleep(1.2)
        
        punish_position = self._calculate_spell_punish_position()
        if len(available_cards) > 1:
            punish_card = available_cards[1]
            hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][punish_card]
            emulator.click(*hand_coord)
            time.sleep(0.1)
            emulator.click(*punish_position)
            
        return True
        
    def _execute_micro_play(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        technique = decision.get("technique", "kiting")
        logger.change_status(f"MICRO TECHNIQUE: {technique}")
        
        if technique in self.micro_techniques:
            return self.micro_techniques[technique](emulator, logger, decision)
            
        return False
        
    def _execute_pig_push(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        logger.change_status("PIG PUSH TECHNIQUE")
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if len(available_cards) < 2:
            return False
            
        building_card = available_cards[0]
        win_con_card = available_cards[1]
        
        building_pos = (400, 350)
        win_con_pos = (450, 320)
        
        hand_coord1 = [(142, 561), (210, 563), (272, 561), (341, 563)][building_card]
        emulator.click(*hand_coord1)
        time.sleep(0.1)
        emulator.click(*building_pos)
        
        time.sleep(0.3)
        
        hand_coord2 = [(142, 561), (210, 563), (272, 561), (341, 563)][win_con_card]
        emulator.click(*hand_coord2)
        time.sleep(0.1)
        emulator.click(*win_con_pos)
        
        return True
        
    def _execute_kiting(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        logger.change_status("KITING MANEUVER")
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if not available_cards:
            return False
            
        kite_card = available_cards[0]
        kite_positions = [(100, 400), (540, 400), (320, 420)]
        
        for pos in kite_positions:
            hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][kite_card]
            emulator.click(*hand_coord)
            time.sleep(0.1)
            emulator.click(*pos)
            time.sleep(0.8)
            
        return True
        
    def _execute_split_push(self, emulator, logger: Logger, decision: Dict[str, any]) -> bool:
        logger.change_status("SPLIT PUSH")
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if len(available_cards) < 2:
            return False
            
        left_card = available_cards[0]
        right_card = available_cards[1]
        
        left_pos = (150, 300)
        right_pos = (490, 300)
        
        hand_coord1 = [(142, 561), (210, 563), (272, 561), (341, 563)][left_card]
        emulator.click(*hand_coord1)
        time.sleep(0.1)
        emulator.click(*left_pos)
        
        time.sleep(0.2)
        
        hand_coord2 = [(142, 561), (210, 563), (272, 561), (341, 563)][right_card]
        emulator.click(*hand_coord2)
        time.sleep(0.1)
        emulator.click(*right_pos)
        
        return True
        
    def _calculate_human_like_delay(self, analysis: Dict[str, any]) -> float:
        base_delay = 0.3
        
        if analysis["defensive_needs"]:
            max_urgency = max(need["urgency"] for need in analysis["defensive_needs"])
            if max_urgency >= 4:
                base_delay = 0.1
            elif max_urgency >= 3:
                base_delay = 0.2
                
        stress_factor = min(1.0, len(analysis.get("defensive_needs", [])) * 0.2)
        base_delay *= (1 - stress_factor)
        
        human_variation = random.uniform(0.05, 0.15)
        
        return max(0.05, base_delay + human_variation)
        
    def _apply_psychological_pressure(self, emulator, logger: Logger, decision: Dict[str, any]) -> None:
        if random.random() < 0.15:
            pressure_technique = random.choice(["well_played", "good_luck", "crying_king"])
            logger.change_status(f"PSYCHOLOGICAL WARFARE: {pressure_technique}")
            
            emote_coord = (67, 521)
            emulator.click(*emote_coord)
            time.sleep(0.2)
            
            emote_positions = [(124, 419), (182, 420), (255, 411), (312, 423)]
            chosen_emote = random.choice(emote_positions)
            emulator.click(*chosen_emote)
            
    def _learn_from_play(self, decision: Dict[str, any], success: bool) -> None:
        play_record = {
            "decision": decision,
            "timestamp": time.time(),
            "success": success,
            "context": self.enemy_tracker.game_state
        }
        
        if success:
            self.battle_memory["successful_plays"].append(play_record)
        else:
            self.battle_memory["failed_plays"].append(play_record)
            
        self._update_opponent_profile(decision, success)
        
    def _update_opponent_profile(self, decision: Dict[str, any], success: bool) -> None:
        action = decision["action"]
        
        if action not in self.battle_memory["opponent_weaknesses"]:
            self.battle_memory["opponent_weaknesses"][action] = {"success": 0, "total": 0}
            
        self.battle_memory["opponent_weaknesses"][action]["total"] += 1
        if success:
            self.battle_memory["opponent_weaknesses"][action]["success"] += 1
            
    def _detect_spell_pattern(self, battle_time: float) -> bool:
        enemy_plays = self.enemy_tracker.game_state.last_enemy_plays
        
        if len(enemy_plays) < 3:
            return False
            
        spell_plays = [play for play in enemy_plays if play.get("type") == "spell"]
        
        if len(spell_plays) >= 2:
            time_diff = spell_plays[-1].get("time", 0) - spell_plays[-2].get("time", 0)
            return 8 <= time_diff <= 12
            
        return False
        
    def _predict_incoming_spell(self) -> str:
        common_spells = ["fireball", "lightning", "rocket", "arrows", "zap"]
        return random.choice(common_spells)
        
    def _detect_advanced_opportunity(self, analysis: Dict[str, any], battle_time: float) -> Optional[Dict[str, any]]:
        if battle_time > 180 and analysis["elixir_advantage"] >= 3:
            return {
                "action": "overtime_push",
                "priority": 5,
                "confidence": 0.95,
                "reasoning": "overtime_advantage"
            }
            
        if self._detect_opponent_cycle_vulnerability():
            return {
                "action": "cycle_break",
                "priority": 4,
                "confidence": 0.85,
                "reasoning": "opponent_cycle_vulnerable"
            }
            
        return None
        
    def _detect_opponent_cycle_vulnerability(self) -> bool:
        recent_plays = list(self.enemy_tracker.game_state.last_enemy_plays)[-4:]
        
        if len(recent_plays) < 4:
            return False
            
        costs = [play.get("cost", 4) for play in recent_plays]
        return sum(costs) <= 12
        
    def _make_tempo_based_decision(self, analysis: Dict[str, any], battle_time: float) -> Dict[str, any]:
        if battle_time < 30:
            return {
                "action": "conservative_opening",
                "priority": 2,
                "confidence": 0.7,
                "reasoning": "early_game_caution"
            }
        elif 30 <= battle_time < 120:
            return {
                "action": "tempo_control",
                "priority": 3,
                "confidence": 0.8,
                "reasoning": "mid_game_control"
            }
        else:
            return {
                "action": "endgame_tactics",
                "priority": 4,
                "confidence": 0.9,
                "reasoning": "late_game_aggression"
            }
            
    def _find_best_counter_card(self, available_cards: List[int], threat: EnemyUnit) -> int:
        counter_priorities = {
            CardType.TANK: [2, 1, 3],
            CardType.SWARM: [3, 0],
            CardType.DPS: [0, 2],
            CardType.SPELL: [1]
        }
        
        preferred_counters = counter_priorities.get(threat.unit_type, available_cards)
        
        for counter in preferred_counters:
            if counter in available_cards:
                return counter
                
        return available_cards[0]
        
    def _calculate_precise_counter_position(self, threat: EnemyUnit, base_pos: Tuple[int, int]) -> Tuple[int, int]:
        threat_x, threat_y = threat.position
        base_x, base_y = base_pos
        
        if threat.unit_type == CardType.TANK:
            pull_distance = 40
            angle = math.atan2(base_y - threat_y, base_x - threat_x)
            precise_x = int(threat_x + pull_distance * math.cos(angle))
            precise_y = int(threat_y + pull_distance * math.sin(angle))
            return (precise_x, precise_y)
        elif threat.unit_type == CardType.SWARM:
            return (base_x + random.randint(-20, 20), base_y + random.randint(-15, 15))
        else:
            return (base_x, base_y)
            
    def _execute_follow_up_defense(self, emulator, logger: Logger, threat: EnemyUnit) -> None:
        if threat.threat_level == ThreatLevel.CRITICAL:
            logger.change_status("FOLLOW-UP DEFENSE")
            time.sleep(1.0)
            
            available_cards = check_which_cards_are_available(emulator, False, True)
            if available_cards:
                support_card = available_cards[0]
                support_pos = (threat.position[0] + 30, threat.position[1] + 40)
                
                hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][support_card]
                emulator.click(*hand_coord)
                time.sleep(0.1)
                emulator.click(*support_pos)
                
    def _calculate_enemy_commitment(self) -> float:
        recent_plays = list(self.enemy_tracker.game_state.last_enemy_plays)[-3:]
        if not recent_plays:
            return 0.0
            
        total_cost = sum(play.get("cost", 4) for play in recent_plays)
        return min(1.0, total_cost / 15.0)
        
    def _select_optimal_counter_lane(self, enemy_commitment: float) -> Lane:
        if enemy_commitment > 0.7:
            committed_lane = self.enemy_tracker.get_most_threatened_lane()
            return Lane.LEFT if committed_lane == Lane.RIGHT else Lane.RIGHT
        else:
            weakest_tower = min(
                self.enemy_tracker.game_state.enemy_tower_health.items(),
                key=lambda x: x[1]
            )[0]
            return Lane.LEFT if weakest_tower == "left" else Lane.RIGHT
            
    def _select_counter_combo(self, available_cards: List[int], lane: Lane) -> List[Tuple[int, float, Tuple[int, int]]]:
        if len(available_cards) < 2:
            return [(available_cards[0], 0.0, self._get_lane_position(lane))]
            
        combo = []
        
        tank_card = available_cards[0]
        support_card = available_cards[1]
        
        tank_pos = self._get_lane_position(lane)
        support_pos = (tank_pos[0] + 40, tank_pos[1] + 30)
        
        combo.append((tank_card, 0.0, tank_pos))
        combo.append((support_card, 1.2, support_pos))
        
        return combo
        
    def _get_lane_position(self, lane: Lane) -> Tuple[int, int]:
        positions = {
            Lane.LEFT: (150, 320),
            Lane.RIGHT: (490, 320),
            Lane.CENTER: (320, 330)
        }
        return positions[lane]
        
    def _execute_counter_support(self, emulator, logger: Logger, lane: Lane) -> None:
        logger.change_status("COUNTER SUPPORT")
        time.sleep(2.0)
        
        available_cards = check_which_cards_are_available(emulator, False, True)
        if available_cards:
            support_card = available_cards[0]
            support_pos = self._get_lane_position(lane)
            support_pos = (support_pos[0], support_pos[1] - 50)
            
            hand_coord = [(142, 561), (210, 563), (272, 561), (341, 563)][support_card]
            emulator.click(*hand_coord)
            time.sleep(0.1)
            emulator.click(*support_pos)


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode: bool,
    fight_mode_choosed: str,
    called_from_launching: bool = False,
    recording_flag: bool = False,
    use_god_tier_ai: bool = True
) -> bool:
    logger.change_status("do_fight_state state")
    logger.change_status("Waiting for battle to start")
    
    if not wait_for_battle_start(emulator, logger):
        logger.change_status("Error waiting for battle to start")
        return False
    
    logger.change_status("Starting fight loop")
    logger.log(f'Fight mode: "{fight_mode_choosed}"')
    
    if use_god_tier_ai and not random_fight_mode:
        battle_engine = MasterBattleEngine()
        success = battle_engine.execute_god_tier_battle(emulator, logger, recording_flag)
    elif random_fight_mode:
        battle_engine = MasterBattleEngine()
        success = battle_engine._execute_random_battle_loop(emulator, logger)
    else:
        success = _fight_loop(emulator, logger, recording_flag, True)
    
    if not success:
        logger.change_status("Failure in fight loop")
        return False
    
    if not called_from_launching:
        if fight_mode_choosed in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_choosed == "Classic 2v2":
            logger.increment_2v2_fights()
    
    time.sleep(10)
    return True


if __name__ == "__main__":
    pass
