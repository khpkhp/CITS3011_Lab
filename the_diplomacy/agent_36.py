# ============================================================
# CITS3011 Project — Group 36
# Agent: StudentAgent
# P2_beta1_base: Pure MCTS (no special features)
# ============================================================

import math
import random
import time
from agent_baselines import Agent
from game import copy_game


# ============================================================
# Global constants
# ============================================================
TIME_LIMIT = 0.85
MAX_ITERATIONS = 25
ROLLOUT_DEPTH = 2
MAX_ROLLOUT_STEPS = 20
EXPLORATION_C = 1.41
NUM_SAMPLES = 4
DEBUG = False

SCORE_TABLE = {
    0: 0.00, 3: 0.05, 5: 0.12, 7: 0.25,
    9: 0.42, 11: 0.60, 14: 0.80,
    16: 0.90, 18: 1.00,
}


def debug(*args):
    if DEBUG:
        print("[DEBUG]", *args)


# ============================================================
# MCTSNode
# ============================================================
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []

    def ucb1(self, total_visits, c=EXPLORATION_C):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration


# ============================================================
# MCTSSearch — Base only
# ============================================================
class MCTSSearch:
    def __init__(self, time_limit=TIME_LIMIT):
        self.time_limit = time_limit
        self.my_power = None

    def search(self, game, my_power):
        self.my_power = my_power

        root = MCTSNode(copy_game(game))
        try:
            root.untried_actions = self._get_full_actions(root.state, my_power)
        except Exception as e:
            debug(f"ROOT ACTIONS ERROR: {e}")
            root.untried_actions = []

        if not root.untried_actions:
            return self._random_orders(game, my_power)

        start = time.perf_counter()
        iterations = 0

        while iterations < MAX_ITERATIONS:
            if time.perf_counter() - start > self.time_limit:
                break
            try:
                node = self._select(root)
                if node.untried_actions:
                    node = self._expand(node)
                result = self._simulate(node.state)
                self._backpropagate(node, result)
                iterations += 1
            except Exception as e:
                debug(f"SEARCH LOOP ERROR: {e}")
                break

        if not root.children:
            return self._random_orders(game, my_power)

        return self._best_action(root)

    def _best_action(self, root):
        if not root.children:
            return []
        return max(root.children, key=lambda c: c.visits).action

    def _select(self, node):
        while node.children and not node.untried_actions:
            total = sum(c.visits for c in node.children)
            node = max(node.children, key=lambda c: c.ucb1(total))
        return node

    def _expand(self, node):
        action = node.untried_actions.pop()
        new_game = copy_game(node.state)
        self._apply_full_action(new_game, action)
        child = MCTSNode(new_game, parent=node, action=action)
        try:
            child.untried_actions = self._get_full_actions(
                new_game, self.my_power
            )
        except Exception as e:
            debug(f"EXPAND ACTIONS ERROR: {e}")
            child.untried_actions = []
        node.children.append(child)
        return child

    def _apply_full_action(self, game, my_orders):
        for power in game.powers.keys():
            if power == self.my_power:
                game.set_orders(power, my_orders)
            else:
                game.set_orders(power, self._random_orders(game, power))
        game.process()

    def _simulate(self, game):
        sim = copy_game(game)
        phases = 0
        while (not sim.is_game_done
               and phases < ROLLOUT_DEPTH
               and phases < MAX_ROLLOUT_STEPS):
            for power in sim.powers.keys():
                sim.set_orders(power, self._random_orders(sim, power))
            sim.process()
            phases += 1
        return self._evaluate(sim)

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    # --------------------------------------------------------
    # Base candidate generation: random action sets
    # --------------------------------------------------------
    def _get_full_actions(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        if not locations:
            return []
        actions = []
        for _ in range(NUM_SAMPLES):
            orders = []
            for loc in locations:
                if possible.get(loc):
                    orders.append(random.choice(possible[loc]))
            if orders:
                actions.append(orders)
        return actions

    def _random_orders(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        orders = []
        for loc in locations:
            if possible.get(loc):
                orders.append(random.choice(possible[loc]))
        return orders

    # --------------------------------------------------------
    # Base evaluation: milestone SC score
    # --------------------------------------------------------
    def _evaluate(self, game):
        my_centers = len(game.powers[self.my_power].centers)
        if game.is_game_done:
            for power in game.powers.keys():
                if len(game.powers[power].centers) >= 18:
                    return 1.0 if power == self.my_power else 0.0
            return min(my_centers / 18.0, 1.0)

        keys = sorted(SCORE_TABLE.keys())
        score = SCORE_TABLE[keys[0]]
        for i in range(1, len(keys)):
            if my_centers <= keys[i]:
                lo, hi = keys[i-1], keys[i]
                frac = (my_centers - lo) / (hi - lo) if hi > lo else 0
                score = SCORE_TABLE[lo] + frac * (SCORE_TABLE[hi] - SCORE_TABLE[lo])
                break
            score = SCORE_TABLE[keys[i]]

        return max(0.0, min(score, 1.0))


# ============================================================
# StudentAgent
# ============================================================
class StudentAgent(Agent):
    def __init__(self, agent_name='Team 36'):
        super().__init__(agent_name)
        self.mcts = MCTSSearch(time_limit=TIME_LIMIT)
        self.game = None
        self.power_name = None

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

    def update_game(self, all_power_orders):
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    def get_actions(self):
        phase = self.game.phase_type
        if phase != 'M':
            return self._fallback_random()
        try:
            orders = self.mcts.search(self.game, self.power_name)
            if not orders:
                orders = self._fallback_random()
        except Exception as e:
            debug(f"MCTS ERROR: {e}")
            orders = self._fallback_random()
        return orders

    def _fallback_random(self):
        possible = self.game.get_all_possible_orders()
        locations = self.game.get_orderable_locations(self.power_name)
        orders = []
        for loc in locations:
            if possible.get(loc):
                orders.append(random.choice(possible[loc]))
        return orders
