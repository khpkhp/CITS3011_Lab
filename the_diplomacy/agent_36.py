# ============================================================
# CITS3011 Project — Group 36
# Agent: StudentAgent
# beta_base_v1.3_safe  (final base)
#
# ALGORITHM FLOW
# ---------------
# get_actions()
#   -> non-movement phase?  -> sanitised random legal orders
#   -> movement phase?      -> MCTS search
#
# MCTS search (time-bounded)
#   -> build fresh root from current game state
#   -> candidates = [greedy, defensive, random, ...]
#   -> loop until TIME_LIMIT:
#        1) select   : descend by UCB1 (adaptive C)
#        2) expand   : pop one candidate, apply to a copy
#        3) simulate : rollout with mixed policy (self=greedy, opp=random),
#                      depth scales with SC count
#        4) backprop : update visits/wins up to the root
#   -> return the action of the most-visited root child
#
# Safety layer
#   -> phase guard prevents non-movement MCTS calls
#   -> order sanitiser filters every returned order
#   -> BFS distance cache cleared each phase
# ============================================================

import math
import random
import time
from collections import deque
from agent_baselines import Agent
from game import copy_game


# ============================================================
# Tunable constants
# ============================================================
TIME_LIMIT           = 0.90
TIME_SAFE_FRACTION   = 0.85
MIN_ITERATIONS       = 10
MAX_ITERATIONS       = 60
MAX_ROLLOUT_STEPS    = 20
EXPLORATION_C        = 1.2
NUM_SAMPLES          = 4
DEBUG                = False

# Feature toggles
USE_TIME_BUDGET   = True
USE_MIXED_ROLLOUT = True
USE_CANDIDATE_MIX = True
USE_ADAPTIVE_C    = True

# Adaptive rollout depth
ADAPTIVE_DEPTH_EARLY  = 1
ADAPTIVE_DEPTH_LATE   = 3
ADAPTIVE_SC_THRESHOLD = 9

# Adaptive exploration schedule (C decays as SCs grow)
C_START    = 1.2
C_END      = 0.8
C_DECAY_SC = 12

# Evaluation table (interpolated)
SCORE_TABLE = {
    0: 0.00, 3: 0.03, 5: 0.10, 7: 0.22,
    9: 0.45, 11: 0.65, 13: 0.80,
    15: 0.90, 17: 0.96, 18: 1.00,
}


# ============================================================
# Debug helper
# ============================================================
def debug(*args):
    if DEBUG:
        print("[DEBUG]", *args)


# ============================================================
# MCTSNode
# Stores one game state, its children, visit/win stats,
# and the list of candidate actions not yet expanded.
# ============================================================
class MCTSNode:
    __slots__ = ("state", "parent", "action", "children",
                 "visits", "wins", "untried_actions")

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []

    # UCB1 balance of exploitation (wins/visits) and exploration
    def ucb1(self, total_visits, c):
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(total_visits + 1) / self.visits)
        return exploit + explore


# ============================================================
# MapGraph
# Army/fleet adjacency built once per game.
# BFS distance helper with per-phase cache.
# ============================================================
class MapGraph:
    def __init__(self, game):
        self.army_adj = {}
        self.fleet_adj = {}
        self.loc_type = {}

        locations = list(game.map.loc_type.keys())

        for loc in locations:
            loc_type = game.map.loc_type[loc]
            loc_upper = loc.upper()
            self.loc_type[loc_upper] = loc_type
            if loc_type in ('LAND', 'COAST'):
                self.army_adj[loc_upper] = set()
            if loc_type in ('WATER', 'COAST'):
                self.fleet_adj[loc_upper] = set()

        for src in locations:
            s = src.upper()
            for dst in locations:
                d = dst.upper()
                if game.map.abuts('A', s, '-', d):
                    self.army_adj.setdefault(s, set()).add(d)
                if game.map.abuts('F', s, '-', d):
                    self.fleet_adj.setdefault(s, set()).add(d)

    # Are two locations adjacent for any unit type?
    def adjacent(self, a, b):
        return (b in self.army_adj.get(a, set()) or
                b in self.fleet_adj.get(a, set()))

    # BFS shortest distance from start to any goal, cached
    def bfs_distance(self, start, goals, graph, cache):
        key = (start, tuple(sorted(goals)))
        if key in cache:
            return cache[key]
        if start not in graph:
            cache[key] = None
            return None
        goals_set = set(g.upper() for g in goals)
        if start in goals_set:
            cache[key] = 0
            return 0
        visited = {start}
        q = deque([(start, 0)])
        result = None
        while q:
            node, dist = q.popleft()
            for nxt in graph.get(node, ()):
                if nxt in visited:
                    continue
                if nxt in goals_set:
                    result = dist + 1
                    break
                visited.add(nxt)
                q.append((nxt, dist + 1))
            if result is not None:
                break
        cache[key] = result
        return result


# ============================================================
# MCTSSearch
# Time-bounded MCTS with heuristic candidates, mixed rollouts,
# adaptive depth, and adaptive exploration constant.
# ============================================================
class MCTSSearch:
    def __init__(self, time_limit=TIME_LIMIT):
        self.time_limit = time_limit
        self.my_power = None
        self.map_graph = None
        self._dist_cache = {}

    # Build the map graph once per game
    def set_map(self, game):
        self.map_graph = MapGraph(game)
        self._dist_cache = {}

    # Clear BFS cache between phases
    def clear_cache(self):
        self._dist_cache = {}

    # --------------------------------------------------------
    # Main MCTS entry point
    # --------------------------------------------------------
    def search(self, game, my_power):
        self.my_power = my_power
        if self.map_graph is None:
            self.set_map(game)

        # Safety: only search during movement phases
        if game.phase_type != 'M':
            return self._random_orders(game, my_power)

        # Fresh root per turn (no cross-phase tree reuse)
        root = MCTSNode(copy_game(game))
        try:
            root.untried_actions = self._get_full_actions(root.state, my_power)
        except Exception as e:
            debug(f"ROOT ACTIONS ERROR: {e}")
            root.untried_actions = []

        if not root.untried_actions:
            return self._random_orders(game, my_power)

        # Set the time budget
        start = time.perf_counter()
        if USE_TIME_BUDGET:
            soft_deadline = start + self.time_limit * TIME_SAFE_FRACTION
            hard_deadline = start + self.time_limit
            iteration_cap = 10 ** 9
        else:
            soft_deadline = float('inf')
            hard_deadline = start + self.time_limit
            iteration_cap = MAX_ITERATIONS

        my_centers = len(game.powers[my_power].centers)
        current_c = self._current_exploration_c(my_centers)

        # Main loop: select -> expand -> simulate -> backprop
        iterations = 0
        while iterations < iteration_cap:
            now = time.perf_counter()
            if now >= hard_deadline:
                break
            if USE_TIME_BUDGET and now >= soft_deadline \
                    and iterations >= MIN_ITERATIONS:
                break

            try:
                node = self._select(root, current_c)
                if node.untried_actions:
                    node = self._expand(node)
                result = self._simulate(node.state, hard_deadline)
                self._backpropagate(node, result)
                iterations += 1
            except Exception as e:
                debug(f"SEARCH LOOP ERROR: {e}")
                break

        debug(f"[v1.3_safe] iterations={iterations}, c={current_c:.2f}, "
              f"elapsed={time.perf_counter()-start:.3f}s")

        if not root.children:
            return self._random_orders(game, my_power)

        return self._best_action(root)

    # --------------------------------------------------------
    # Adaptive exploration constant
    # --------------------------------------------------------
    def _current_exploration_c(self, my_centers):
        if not USE_ADAPTIVE_C:
            return EXPLORATION_C
        if my_centers >= C_DECAY_SC:
            return C_END
        frac = my_centers / C_DECAY_SC
        return C_START + frac * (C_END - C_START)

    # --------------------------------------------------------
    # Pick the most-visited child (tie-break by win rate)
    # --------------------------------------------------------
    def _best_action(self, root):
        if not root.children:
            return []
        return max(
            root.children,
            key=lambda c: (c.visits, c.wins / max(c.visits, 1))
        ).action

    # --------------------------------------------------------
    # Select: descend until a node with untried actions
    # --------------------------------------------------------
    def _select(self, node, c):
        while node.children and not node.untried_actions:
            total = sum(ch.visits for ch in node.children)
            node = max(node.children, key=lambda ch: ch.ucb1(total, c))
        return node

    # --------------------------------------------------------
    # Expand: pop one candidate, apply it to a copy
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Apply my orders and random orders for all other powers
    # --------------------------------------------------------
    def _apply_full_action(self, game, my_orders):
        for power in game.powers.keys():
            if power == self.my_power:
                game.set_orders(power, my_orders)
            else:
                game.set_orders(power, self._random_orders(game, power))
        game.process()

    # --------------------------------------------------------
    # Simulate: rollout from a node's state to a depth limit
    # --------------------------------------------------------
    def _simulate(self, game, hard_deadline):
        sim = copy_game(game)
        my_centers = len(sim.powers[self.my_power].centers)

        # Depth selection
        if my_centers >= ADAPTIVE_SC_THRESHOLD:
            depth = ADAPTIVE_DEPTH_LATE
        else:
            depth = ADAPTIVE_DEPTH_EARLY

        # Shrink if close to the deadline
        if hard_deadline - time.perf_counter() < 0.05:
            depth = 1

        phases = 0
        while (not sim.is_game_done
               and phases < depth
               and phases < MAX_ROLLOUT_STEPS):
            if time.perf_counter() >= hard_deadline:
                break
            for power in sim.powers.keys():
                if USE_MIXED_ROLLOUT and power == self.my_power:
                    # My side plays greedy during rollouts
                    sim.set_orders(power, self._greedy_orders(sim, power))
                else:
                    # Opponents play random
                    sim.set_orders(power, self._random_orders(sim, power))
            sim.process()
            phases += 1
        return self._evaluate(sim)

    # --------------------------------------------------------
    # Backpropagate the rollout value up to the root
    # --------------------------------------------------------
    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    # --------------------------------------------------------
    # Candidate generation at a node
    # --------------------------------------------------------
    def _get_full_actions(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        if not locations:
            return []

        actions = []
        signatures = set()

        # Deduplicate before adding
        def _add(orders):
            if not orders:
                return
            sig = tuple(sorted(orders))
            if sig in signatures:
                return
            signatures.add(sig)
            actions.append(orders)

        # Heuristic candidates first
        if USE_CANDIDATE_MIX:
            try:
                _add(self._greedy_orders(game, power))
            except Exception as e:
                debug(f"GREEDY CANDIDATE ERROR: {e}")
            try:
                _add(self._defensive_orders(game, power))
            except Exception as e:
                debug(f"DEFENSIVE CANDIDATE ERROR: {e}")

        # Fill remainder with random candidates
        guard = 0
        while len(actions) < NUM_SAMPLES and guard < NUM_SAMPLES * 4:
            guard += 1
            orders = []
            for loc in locations:
                if possible.get(loc):
                    orders.append(random.choice(possible[loc]))
            _add(orders)

        return actions[:NUM_SAMPLES]

    # --------------------------------------------------------
    # Heuristic candidate: move every unit toward nearest SC
    # --------------------------------------------------------
    def _greedy_orders(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        if not locations:
            return []
        my_centers = set(c.upper() for c in game.get_centers(power))
        enemy_centers = [c.upper() for c in game.map.scs
                         if c.upper() not in my_centers]
        if not enemy_centers:
            return self._random_orders(game, power)

        orders = []
        for loc in locations:
            legal = possible.get(loc, [])
            if not legal:
                continue
            is_army = any(o.startswith('A ') for o in legal)
            graph = (self.map_graph.army_adj if is_army
                     else self.map_graph.fleet_adj)
            best_order, best_dist = None, None
            for order in legal:
                parts = order.split()
                if len(parts) >= 4 and parts[2] == '-':
                    t = parts[3].upper()
                    d = self.map_graph.bfs_distance(
                        t, enemy_centers, graph, self._dist_cache
                    )
                    if d is not None and (best_dist is None or d < best_dist):
                        best_dist = d
                        best_order = order
            if best_order is not None:
                orders.append(best_order)
            else:
                holds = [o for o in legal if o.endswith(' H')]
                orders.append(random.choice(holds) if holds
                              else random.choice(legal))
        return orders

    # --------------------------------------------------------
    # Heuristic candidate: hold home SCs, others move back
    # --------------------------------------------------------
    def _defensive_orders(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        if not locations:
            return []
        my_centers = set(c.upper() for c in game.get_centers(power))
        orders = []
        for loc in locations:
            legal = possible.get(loc, [])
            if not legal:
                continue
            if loc.upper() in my_centers:
                holds = [o for o in legal if o.endswith(' H')]
                if holds:
                    orders.append(random.choice(holds))
                    continue
            is_army = any(o.startswith('A ') for o in legal)
            graph = (self.map_graph.army_adj if is_army
                     else self.map_graph.fleet_adj)
            best_order, best_dist = None, None
            for order in legal:
                parts = order.split()
                if len(parts) >= 4 and parts[2] == '-':
                    t = parts[3].upper()
                    d = self.map_graph.bfs_distance(
                        t, list(my_centers), graph, self._dist_cache
                    )
                    if d is not None and (best_dist is None or d < best_dist):
                        best_dist = d
                        best_order = order
            if best_order is not None:
                orders.append(best_order)
            else:
                holds = [o for o in legal if o.endswith(' H')]
                orders.append(random.choice(holds) if holds
                              else random.choice(legal))
        return orders

    # --------------------------------------------------------
    # Random legal orders for a power
    # --------------------------------------------------------
    def _random_orders(self, game, power):
        possible = game.get_all_possible_orders()
        locations = game.get_orderable_locations(power)
        orders = []
        for loc in locations:
            legal = possible.get(loc, [])
            if legal:
                orders.append(random.choice(legal))
        return orders

    # --------------------------------------------------------
    # Evaluation: interpolated SCORE_TABLE
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
                lo, hi = keys[i - 1], keys[i]
                frac = (my_centers - lo) / (hi - lo) if hi > lo else 0
                score = SCORE_TABLE[lo] + frac * (
                    SCORE_TABLE[hi] - SCORE_TABLE[lo]
                )
                break
            score = SCORE_TABLE[keys[i]]

        return max(0.0, min(score, 1.0))


# ============================================================
# StudentAgent
# Public interface used by the game engine.
# ============================================================
class StudentAgent(Agent):
    def __init__(self, agent_name='Team 36'):
        super().__init__(agent_name)
        self.mcts = MCTSSearch(time_limit=TIME_LIMIT)
        self.game = None
        self.power_name = None

    # Called once per game by the engine
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.mcts.set_map(game)

    # Called after every phase to keep state consistent
    def update_game(self, all_power_orders):
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()
        self.mcts.clear_cache()

    # Called every phase where we must submit orders
    def get_actions(self):
        phase = self.game.phase_type

        # Non-movement: sanitised random legal orders
        if phase != 'M':
            return self._sanitise_orders(self._fallback_random())

        # Movement: run MCTS, then sanitise the result
        try:
            raw = self.mcts.search(self.game, self.power_name)
        except Exception as e:
            debug(f"MCTS ERROR: {e}")
            raw = self._fallback_random()

        clean = self._sanitise_orders(raw)
        if clean:
            return clean
        return self._sanitise_orders(self._fallback_random())

    # --------------------------------------------------------
    # Drop any order not in the current legal set
    # --------------------------------------------------------
    def _sanitise_orders(self, orders):
        if not orders:
            return []
        possible = self.game.get_all_possible_orders()
        orderable = self.game.get_orderable_locations(self.power_name)

        legal_pool = set()
        for loc in orderable:
            for o in possible.get(loc, []):
                legal_pool.add(o)

        clean = []
        for o in orders:
            if not isinstance(o, str):
                continue
            if o in legal_pool:
                clean.append(o)
        return clean

    # --------------------------------------------------------
    # Fallback: one random legal order per orderable location
    # --------------------------------------------------------
    def _fallback_random(self):
        possible = self.game.get_all_possible_orders()
        locations = self.game.get_orderable_locations(self.power_name)
        orders = []
        for loc in locations:
            legal = possible.get(loc, [])
            if legal:
                orders.append(random.choice(legal))
        return orders
