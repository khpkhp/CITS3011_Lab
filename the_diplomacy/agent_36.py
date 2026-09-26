import time
import timeout_decorator
import random
import game
import networkx as nx
import itertools
import math

'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from agent_baselines import Agent, GreedyAgent, StaticAgent, AttitudeAgent


class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''
    #random funcs
    def get_loc_neighbors(self, loc): #quick note: this is building a graph regardless of unit type, is causing bugs down there will work on this later
        neighbors = set()
        loc_upper = loc.upper()
        if loc_upper in self.map_graph_army:
            neighbors.update(self.map_graph_army.neighbors(loc_upper))
        if loc_upper in self.map_graph_navy:
            neighbors.update(self.map_graph_navy.neighbors(loc_upper))
        return neighbors

    def get_units_neighbors(self, loc): #graph for different unit types, still buggy tho since there r certain adjacent coastal nodes fleets cant actually move betweek (VEN-ROM) VERY PROBLEMATIC FOR ITALY
        army = self.power.units
        neighbors = set()
        loc_upper = loc.upper()
        type = ''
        for units in army:
            unit_type, unit_loc = units.split(' ')
            if unit_loc.split('/')[0] == loc_upper:
                type = unit_type
                break
        if type == 'F':
            if loc_upper in self.map_graph_navy:
                neighbors.update(self.map_graph_navy.neighbors(loc_upper))
        else:
            if loc_upper in self.map_graph_army:
                neighbors.update(self.map_graph_army.neighbors(loc_upper))
        return (neighbors, type)

    def get_furthest_neighbor(self, game_state, loc_neighbors, loc_type):
        army = self.power.units
        my_home = self.power.homes
        game_map = game_state.map
        best_neighbor = None

        if loc_type == 'A':
            closest_home = None
            furthest_dist = -1
            for neighbor in loc_neighbors:
                min_dist_from_home = 100000
                if neighbor in self.map_graph_army:
                    paths = nx.shortest_path_length(self.map_graph_army, source=neighbor)
                else:
                    continue
                for home in my_home:
                    if home in paths and paths[home] < min_dist_from_home:
                        min_dist_from_home = paths[home]
                        closest_home = home
                if min_dist_from_home > furthest_dist:
                    best_neighbor = neighbor
                    furthest_dist = min_dist_from_home
        else:
            closest_home = None
            furthest_dist = -1
            for neighbor in loc_neighbors:
                min_dist_from_home = 100000
                if neighbor in self.map_graph_navy:
                    paths = nx.shortest_path_length(self.map_graph_navy, source=neighbor)
                else: 
                    continue
                for home in my_home:
                    if home in paths and paths[home] < min_dist_from_home:
                        min_dist_from_home = paths[home]
                        closest_home = home
                if min_dist_from_home > furthest_dist:
                    best_neighbor = neighbor
                    furthest_dist = min_dist_from_home
        return best_neighbor
    
    def get_target(self, str): #just a func to get the target loc of a move 
        move = str.split(' ')
        if '-' in move:
            return move[move.index('-') + 1]
        return None

    #custom USEFUL function up here
    def build_map_graphs(self):    # reuse func from greedy agent
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys()) # locations with '/' are not real provinces

        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())

        locations = [i.upper() for i in locations]

        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)
        self.greedy_agent.map_graph_army = self.map_graph_army
        self.greedy_agent.map_graph_navy = self.map_graph_navy

    def brexit(self, game_state, loc):
        my_units = game_state.get_orderable_locations(self.power_name)
        final_dest = None
        fleet_num = 99
        fleet_req = set()
        for dest, fleet_options in game_state.convoy_paths_dest.get(loc).items():
            if dest not in ['WAL', 'LON'] and dest not in my_units:
                for f in fleet_options:
                    if len(f) < fleet_num:
                        fleet_num = len(f)
                        fleet_req = f
                        final_dest = dest
        return (final_dest, fleet_req)



    def evaluate_pairs(self, game_state, pairs_list, pairs_num): #rationally rank pairs to reduce number of pairs need to be evaluated
        game_map = game_state.map
        my_scs = set(game_state.get_centers(self.power_name))
        enemy_scs = set(game_map.scs) - my_scs
        influence = set(self.power.influence)
        my_units = set(game_state.get_orderable_locations(self.power_name))
        enemy_powers = [p for p in game_state.get_map_power_names() if p != self.power_name]
        enemy_unit_locs = set()
        for opp in enemy_powers:
            enemy_unit_locs.update(game_state.get_orderable_locations(opp))

        all_units_locs = my_units | enemy_unit_locs
        unoccupied_scs = enemy_scs - all_units_locs


        scored_pairs = []
        for pairs in pairs_list:
            pair_score = 0.0
            #this is the most useless province in the whole game, pls dont go here
            if pairs[0] == "SYR" or pairs[1] == "SYR":
                pair_score -= 10000000000000.0
            #if ur gonna attack 2 targets next to each other, just attack 1 bro
            if pairs[0] in self.get_units_neighbors(pairs[1])[0]:
                pair_score += -100.0
            #extend, dont waste time attacking our own provinces
            if pairs[0] in influence or pairs[1] in influence:
                pair_score += -50000.0
            #everything else should be self-explainatory
            pair_score -= 1000**(self.target_history.count(pairs))  
            pair_score += len(self.get_units_neighbors(pairs[0])[0] & my_units) * 50
            pair_score += len(self.get_units_neighbors(pairs[1])[0] & my_units) * 20
            if pairs[0] in unoccupied_scs:
                pair_score += 1000
            if pairs[0] in enemy_scs:
                pair_score += 500
            if pairs[1] in unoccupied_scs:
                pair_score += 100
            if pairs[1] in enemy_scs:
                pair_score += 50
            scored_pairs.append((pair_score, pairs))
        scored_pairs.sort(key =lambda x: x[0], reverse= True)
        return [pair for (pair_score, pair) in scored_pairs[:pairs_num]]

    def evaluate(self, game_state):    #heuristic? to evaluate board positions
        game_map = game_state.map    
        my_scs = set(game_state.get_centers(self.power_name))
        enemy_scs = set(game_map.scs) - my_scs

        #my unit locations
        my_units = set(game_state.get_orderable_locations(self.power_name))

        #enemy unit locations
        enemy_powers = [p for p in game_state.get_map_power_names() if p != self.power_name]
        enemy_unit_locs = set()
        for opp in enemy_powers:
            enemy_unit_locs.update(game_state.get_orderable_locations(opp))

        all_units_locs = my_units | enemy_unit_locs
        unoccupied_scs = enemy_scs - all_units_locs

        #base SC Score
        scs_score = len(my_scs) * 1000.0

        #SC Pressure
        scs_pressure_score = 0
        for loc in my_units:
            adj_locs, type = self.get_units_neighbors(loc)
            empty_scs_num = len(adj_locs & unoccupied_scs)
            scs_pressure_score += (empty_scs_num * 100.0)

            defended_scs_num = len(adj_locs & (enemy_scs & enemy_unit_locs))
            scs_pressure_score += (defended_scs_num * 50.0) 

        #mutual Unit Support Range
        mutual_support_score = 0
        for loc in my_units:
            adj_locs, type = self.get_units_neighbors(loc)
            mutual_support_score += 20.0 * (len(adj_locs & my_units))

        #threat Penalty (Enemy units adjacent to our SCs)
        threat_penalty = 0.0
        for sc in my_scs:
            sc_adj = self.get_loc_neighbors(sc)
            threat_count = len(sc_adj & enemy_unit_locs)
            threat_penalty += (threat_count * 100.0)

        strat_bonus = 0.0
        if self.power_name == 'ENGLAND':
            if game_state.phase_type == 'A':
                for loc in my_units:
                    adj_locs, type = self.get_units_neighbors(loc)
                    if loc in ['EDI', 'LON'] and type == 'A':
                        strat_bonus += 100
            if 'NTH' in my_units:
                strat_bonus += 5000
            if 'ENG' in my_units:
                strat_bonus += 5000





        return scs_score + scs_pressure_score + mutual_support_score - threat_penalty + strat_bonus
    
    def focus_target(self, game_state, mult_target_loc): #refer to docs. focus all troops on 2 provinces
        possible_orders = game_state.get_all_possible_orders()
        orderable_locs = game_state.get_orderable_locations(self.power_name)
        orderable_locs.sort(key=lambda loc: self.get_units_neighbors(loc)[1])
        game_map = game_state.map    
        my_scs = set(game_state.get_centers(self.power_name))
        enemy_scs = set(game_map.scs) - my_scs
        my_units = set(game_state.get_orderable_locations(self.power_name))

        #enemy unit locations
        enemy_powers = [p for p in game_state.get_map_power_names() if p != self.power_name]
        enemy_unit_locs = set()
        for opp in enemy_powers:
            enemy_unit_locs.update(game_state.get_orderable_locations(opp))
        all_units_locs = my_units | enemy_unit_locs
        unoccupied_scs = enemy_scs - all_units_locs

        if not orderable_locs:
            return []

        final_orders = []
        assigned_units = set() #use set to avoid duplicates
        mult_target_loc = [target.split('/')[0] for target in mult_target_loc] 
        #case: defend; if target is alr occupied by friendly troop
        for target_loc in mult_target_loc:
            #case: attack
            #else:
                attackers = []
                is_convoy = False
                for loc in orderable_locs:
                    if loc not in assigned_units:
                        for order in possible_orders.get(loc, []):
                            if self.get_target(order) == target_loc and ' S ' not in order and ' C ' not in order:
                                attackers.append((loc, order))
                #select a primary attacker
                #FUCK WHY IS THIS ATTACKER LIST ALWAYS EMPTY SOME1 PLS FUCKING HELP ITS 2AM oh shit it took me a day i got it now
                if attackers:
                    # convoy_attacks = [a for a in attackers if 'VIA' in a[1]]            
                    # normal_attacks = [a for a in attackers if 'VIA' not in a[1]]
                    # if convoy_attacks: #always prio convoy attacks
                    #     primary_loc, primary_order = random.choice(convoy_attacks)
                    #     is_convoy = True
                    # else:    
                    #     primary_loc, primary_order = random.choice(normal_attacks)
                    attackers_army = [unit for unit in attackers if unit[1].startswith("A")]
                    if attackers_army:
                        primary_loc, primary_order = random.choice(attackers_army)
                    else:
                        primary_loc, primary_order = random.choice(attackers)
                    if 'VIA' in primary_order:
                        is_convoy = True
                    final_orders.append(primary_order)
                    assigned_units.add(primary_loc)

                    core_order = primary_order.replace(' VIA', '') if is_convoy else primary_order #in case the attack is via convoy -> strip the convoy part since the format for moves supporting convoy doesnt have via
                    #adjacent units supporting/convoying primary attacker
                    for loc in orderable_locs:
                        if loc not in assigned_units:
                            convoy_order = []
                            sp_order = []
                            for order in possible_orders.get(loc, []): #look thru all orders to find convoy orders
                                if is_convoy and ' C ' in order and order.endswith(f' {core_order}'):
                                        convoy_order.append(order)
                                        break
                                elif ' S ' in order and order.endswith(f' {core_order}'):
                                        sp_order.append(order)
                            if convoy_order: #prio convoy order, if attacking via convoy ALL FLEET AVAIL MUST CONVOY
                                final_orders.append(convoy_order[0])
                                assigned_units.add(loc)
                            elif sp_order: #otherwise sp the convoy
                                final_orders.append(sp_order[0])
                                assigned_units.add(loc)
                            

        #case: hold; for the moment, all remaining troops just hold lol im losing my sanity
        #now will move to nearby unoccupied scs, then neighboring tile to target
        common_target = set()
        convoy_target = set()
        fleets_req = set()
        convoy_path = None
        brexited = False
        for loc in orderable_locs:
            if loc not in assigned_units:
                neighbors, type = self.get_units_neighbors(loc)
                loc_neighbors = neighbors - set(orderable_locs + ["SYR", 'APU']) 
                target1_loc_neighbor = self.get_loc_neighbors(mult_target_loc[0])
                target2_loc_neighbor = self.get_loc_neighbors(mult_target_loc[1])
                scs_unoccupied_neighbors = loc_neighbors & unoccupied_scs
                scs_occupied_neighbors = loc_neighbors & enemy_scs
                target = []
                primary_order = []
                
                #gtfo of britain
                if loc in ["LON", 'EDI', 'YOR', 'WAL'] and self.power_name == "ENGLAND" and not brexited and type == 'A':
                    if game_state.convoy_paths_dest.get(loc):
                        dest, fleets_req = self.brexit(game_state, loc)
                        convoy_target.add(dest)
                        final_orders.append(f'A {loc} - {dest} VIA')
                        convoy_path = f'A {loc} - {dest}'
                        brexited = True
                        assigned_units.add(loc)
                        continue
                elif convoy_target and loc in fleets_req:
                    final_orders.append(f'F {loc} C {convoy_path}') 
                    assigned_units.add(loc)           
                    continue
                #check if theres alr someone attacking this target first -> sp them, also prioritize common attacks than individual attacks
                elif convoy_target & loc_neighbors:
                    dest = random.choice(list(convoy_target & loc_neighbors))
                    for orders in final_orders: 
                        order_txt = orders.split()
                        if ' S ' not in orders and order_txt[-2] == dest:
                            primary_order.append(' '.join(order_txt[:-1]))
                
                elif loc_neighbors & common_target:
                    target.append(random.choice(list(loc_neighbors & common_target)))
                    for order in final_orders:
                        if ' S ' not in order and f' - {target[0]}' in order:
                            primary_order.append(order)
                            break
                elif self.power_name == "ENGLAND" and (loc == "ENG" or loc == "NTH"):
                    pass
                #priority list for attacks: unoccupied scs -> occupied scs -> nearby units
                elif loc in enemy_scs:
                    assigned_units.add(loc)
                    continue
                elif scs_unoccupied_neighbors:
                    target.append(random.choice(list(scs_unoccupied_neighbors)))
                elif scs_occupied_neighbors:
                    target.append(random.choice(list(scs_occupied_neighbors)))
                elif target1_loc_neighbor & loc_neighbors:
                    target.append(random.choice(list(target1_loc_neighbor & loc_neighbors)))
                elif target2_loc_neighbor & loc_neighbors:
                    target.append(random.choice(list(target2_loc_neighbor & loc_neighbors)))    
                elif loc_neighbors:
                    # target.append(random.choice(list(loc_neighbors)))
                    target.append(self.get_furthest_neighbor(game_state, loc_neighbors, type))                

                if target or primary_order:
                    if target:
                        common_target.add(target[0])
                    #if theres a primary attack -> support
                    if primary_order:
                        for order in possible_orders.get(loc, []):
                            if ' S ' in order and order.endswith(f' {primary_order[0]}'):
                                final_orders.append(order)
                                assigned_units.add(loc)
                                break
                    else:   #otherwise attack on ur own
                        for order in possible_orders.get(loc, []):
                            if self.get_target(order) == target[0]  and ' S ' not in order and ' C ' not in order and 'VIA' not in order:
                                final_orders.append(order)
                                assigned_units.add(loc)
                                break
                else:   #otherwise hold
                    order = [o for o in possible_orders.get(loc, []) if o.endswith(' H')][0]
                    final_orders.append(order)
        return (final_orders, mult_target_loc)

    def gen_candidate_target(self, game_state, num_targets=5): #generate random targets to focus. currently prioritising attacking and only defending scs
        orderable_locs = game_state.get_orderable_locations(self.power_name)
        if not orderable_locs:
            return []

        possible_orders = game_state.get_all_possible_orders()

        #get all targets avail 
        defense_targets = set(game_state.get_centers(self.power_name)) & set(orderable_locs)
        attack_targets = set()
        #find all possible locs for attacks
        # for loc in orderable_locs:
        #     for order in possible_orders.get(loc, []):
        #         if ' S ' not in order and ' C ' not in order:
        #             target = self.get_target(order)
        #             if target and target not in defense_targets and target in self.get_units_neighbors:
        #                 attack_targets.add(target)
        # reachable_targets = defense_targets | attack_targets

        for loc in orderable_locs:
            attack_targets.update(self.get_units_neighbors(loc)[0])
        cleaned_attack_targets = {loc.split('/')[0] for loc in attack_targets}

        influence = set(self.power.influence)
        attack_pool = cleaned_attack_targets - influence
        targets = list(attack_pool)
        defense_pool = list(defense_targets)
        #combine to make a 2d list (its a duo mcts)
        mult_attack_pool = [list(plan) for plan in itertools.combinations(targets, 2)]
        candidate_targets = self.evaluate_pairs(game_state, mult_attack_pool, num_targets)

        #solely taking attacking locs tho -> extremely greedy
        if not attack_pool:
            return []

        # if len(mult_attack_pool) < num_targets:
        #     candidate_targets += mult_attack_pool
        #     for _ in range(num_targets * 3):
        #         if defense_pool:
        #             target = [random.choice(attack_pool), random.choice(defense_pool)]
        #             if target not in candidate_targets:
        #                 candidate_targets.append(target)
        #                 if len(candidate_targets) >= num_targets:
        #                     break
        # else:
        #     for _ in range(num_targets * 3):
        #         target = random.choice(mult_attack_pool)
        #         if target not in candidate_targets:
        #             candidate_targets.append(target)
        #             if len(candidate_targets) >= num_targets:
        #                 break
        return candidate_targets

    #NOTICE! CURRENTLY USING GREEDY BASELINE AGENT TO SIM OPPONENTS. NEED TO CHECK IF THIS IS ALLOWED        
    def get_action_opp(self, game_state, opp_power):
        orders_history = game_state.order_history.get('S1901M', {})
        if orders_history is None:
            agent = self.greedy_agent
        elif len(orders_history.get(opp_power, [])) == 0:
            agent = self.static_agent
        else:
            agent = self.greedy_agent
        agent.game = game_state
        agent.power_name = opp_power
        return agent.get_actions()
  

    #base mcts search, depth 1
    def mcts_search(self, game_state, candidate, sims_per_plan):
        if not candidate:   #this should NEVER be used pls 
            my_units = game_state.get_orderable_locations(self.power_name)
            possible_orders = game_state.get_all_possible_orders()
            return [possible_orders[loc][0] for loc in my_units if possible_orders.get(loc)]

        plan_scores = {i: 0.0 for i in range(len(candidate))}
        opponent_powers = [p for p in game_state.get_map_power_names() if p != self.power_name]

        for idx, (my_orders, target) in enumerate(candidate):
            for _ in range(sims_per_plan):
                sim_game = game.copy_game(game_state)
                orders = {self.power_name: my_orders}

                for opp in opponent_powers:
                    orders[opp] = self.get_action_opp(sim_game, opp)

                try:
                    for p, o in orders.items():
                        sim_game.set_orders(p, o)
                    sim_game.process()
                except Exception:
                    continue

                score = self.evaluate(sim_game)
                plan_scores[idx] += score

        best_plan = max(plan_scores.keys(), key=lambda i: plan_scores[i])
        final_orders, final_targets = candidate[best_plan]
        # print(final_targets, self.target_history.count(final_targets))
        self.target_history.append(final_targets)
        return final_orders

    def make_orders(self, game_state):
        if self.game.phase_type != 'M':
            possible_orders = game_state.get_all_possible_orders()
            my_locs = game_state.get_orderable_locations(self.power_name)
            #balance num of fleet and army: build fleet/army first, then at least 1 of the other for each build phase
            if self.power_name == "ENGLAND": #build fleet first if england
                if self.game.phase_type == 'A' and self.navy_num <= 4:
                    options_per_loc = [[order for order in possible_orders[loc] if order.startswith('F')] for loc in my_locs if possible_orders.get(loc)]
                else:
                    options_per_loc = [possible_orders[loc] for loc in my_locs if possible_orders.get(loc)]
                if not options_per_loc:
                    return []
                if self.navy_num > 4:
                    candidate_orders = [(list(plan), "empty") for plan in itertools.product(*options_per_loc) if any(p.startswith('A') for p in plan)]
                else: 
                    candidate_orders = [(list(plan), "empty") for plan in itertools.product(*options_per_loc)]

            elif self.power_name == "ITALY": 
                if self.game.phase_type == 'A' and self.army_num <= 4:
                    options_per_loc = [[order for order in possible_orders[loc] if order.startswith('A')] for loc in my_locs if possible_orders.get(loc)]
                else:
                    options_per_loc = [possible_orders[loc] for loc in my_locs if possible_orders.get(loc)]

                candidate_orders = [(list(plan), "empty") for plan in itertools.product(*options_per_loc)]

            else:   #build army first if anything else
                if self.game.phase_type == 'A' and self.army_num <= 4:
                    options_per_loc = [[order for order in possible_orders[loc] if order.startswith('A')] for loc in my_locs if possible_orders.get(loc)]
                else:
                    options_per_loc = [possible_orders[loc] for loc in my_locs if possible_orders.get(loc)]

                candidate_orders = [(list(plan), "empty") for plan in itertools.product(*options_per_loc)]


            #mcts and evaluate
            #print(time.perf_counter() - start_time, self.game.phase_type)
            return self.mcts_search(game_state, candidate_orders, sims_per_plan=1)
        elif self.power_name == "ENGLAND" and game_state.phase == "SPRING 1901 MOVEMENT":
            return ['F EDI - NWG', 'F LON - NTH', 'A LVP - EDI']
        elif self.power_name == "ENGLAND" and game_state.phase == "FALL 1901 MOVEMENT":
            return ['F NTH - HOL', 'F NWG C A EDI - NWY', 'A EDI - NWY VIA']
        elif self.power_name == "ITALY" and game_state.phase == "SPRING 1901 MOVEMENT":
            return ['F NAP - ION', 'A VEN - TYR', 'A ROM - VEN']
        elif self.power_name == "ITALY" and game_state.phase == "FALL 1901 MOVEMENT":
            return ['F ION - GRE', 'A VEN - TRI', 'A TYR S A VEN - TRI']
        elif self.power_name == "FRANCE" and game_state.phase == "SPRING 1901 MOVEMENT":
            return ['F BRE - MAO', 'A PAR - PIC', 'A MAR H']
        elif self.power_name == "FRANCE" and game_state.phase == "FALL 1901 MOVEMENT":
            return ['F MAO - POR', 'A PIC - BEL', 'A MAR - SPA']
        elif self.power_name == "TURKEY" and game_state.phase == "SPRING 1901 MOVEMENT":
            return ['A CON - BUL', 'F ANK - BLA', 'A SMY - ARM']

        else:
            candidate_targets = self.gen_candidate_target(game_state)
            candidate_orders = []
            existing_orders = []

            for target in candidate_targets:
                (plan, intended_target) = self.focus_target(game_state, target)
                
                if plan and plan not in existing_orders:
                    existing_orders.append(plan)
                    candidate_orders.append((plan, intended_target))
            return self.mcts_search(self.game, candidate_orders, sims_per_plan=1)

    #@timeout_decorator.timeout(1)
    def __init__(self, agent_name='im_horrible_at_giving_nicknames'):
        super().__init__(agent_name)
        self.greedy_agent = GreedyAgent('GreedySim')
        self.static_agent = StaticAgent('StaticSim')
        '''Implement your agent here.'''

    #@timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.target_history = []
        self.army_num = 2
        self.navy_num = 1
        self.power = game.get_power(self.power_name)
        if power_name == "RUSSIA":
            self.navy_num = 2
        if power_name == "ENGLAND":
            self.army_num = 1
            self.navy_num = 2
        self.build_map_graphs()

        '''Implement your agent here.'''

    #@timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])

        self.army_num = len([unit for unit in self.power.units if unit.startswith("A")])
        self.navy_num = len(self.power.units) - self.army_num
        self.game.process()



    #@timeout_decorator.timeout(1)
    def get_actions(self):
        start_time = time.perf_counter()
        orders = self.make_orders(self.game)
        
        # print(time.perf_counter() - start_time, self.game.phase)
        # print(self.army_num, self.navy_num)
            
        return orders

        '''
        Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        Expected format:
        A LON H                  # Army at LON holds
        F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        A WAL R LON              # Army at WAL retreats to LON
        A LON D                  # Disband Army at LON
        A LON B                  # Build Army at LON
        F EDI B                  # Build Fleet at EDI

        Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        Note: For each unit, if no order is given, it will 'hold' by default.

        Useful Functions:
        
        # This is a dict of all the possible orders for each unit at each location (for all powers).
        possible_orders = self.game.get_all_possible_orders()

        # This is a list of all orderable locations for the power you control.
        orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # Combining these two, you can have the full action space for the power you control.

        # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        '''
