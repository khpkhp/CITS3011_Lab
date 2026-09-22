import time
import timeout_decorator
import random
import game
import networkx as nx
import itertools

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
from agent_baselines import Agent, GreedyAgent, StaticAgent


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

    def get_units_neighbors(self, loc):
        army = self.power.units
        neighbors = set()
        loc_upper = loc.upper()
        type = ''
        for units in army:
            unit_type, unit_loc = units.split(' ')
            if unit_loc == loc_upper:
                type = unit_type
                break
        if type == 'F':
            if loc_upper in self.map_graph_navy:
                neighbors.update(self.map_graph_navy.neighbors(loc_upper))
        else:
            if loc_upper in self.map_graph_army:
                neighbors.update(self.map_graph_army.neighbors(loc_upper))
        return neighbors

    
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
        self.greedy_opp_agent.map_graph_army = self.map_graph_army
        self.greedy_opp_agent.map_graph_navy = self.map_graph_navy
        
    def evaluate_pairs(self, game_state, pairs_list, pairs_num):
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
            if pairs[0] in self.get_loc_neighbors(pairs[1]):
                pair_score += -50.0
            if pairs[0] in influence or pairs[1] in influence:
                pair_score += -100.0
            pair_score += len(self.get_loc_neighbors(pairs[0]) & my_units) * 20
            pair_score += len(self.get_loc_neighbors(pairs[1]) & my_units) * 20
            if pairs[0] in unoccupied_scs:
                pair_score += 100
            if pairs[0] in enemy_scs:
                pair_score += 20
            if pairs[1] in unoccupied_scs:
                pair_score += 100
            if pairs[1] in enemy_scs:
                pair_score += 20
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

        # 1. Base SC Score
        scs_score = len(my_scs) * 1000.0

        # 2. Fast Offensive SC Pressure
        scs_pressure_score = 0
        for loc in my_units:
            adj_locs = self.get_units_neighbors(loc)
            empty_scs_num = len(adj_locs & unoccupied_scs)
            scs_pressure_score += (empty_scs_num * 100.0)

            defended_scs_num = len(adj_locs & (enemy_scs & enemy_unit_locs))
            scs_pressure_score += (defended_scs_num * 50.0) 

        # 3. Mutual Unit Support Range
        mutual_support_score = 0
        for loc in my_units:
            adj_locs = self.get_units_neighbors(loc)
            mutual_support_score += 20.0 * (len(adj_locs & my_units))

        # 4. Threat Penalty (Enemy units adjacent to our SCs)
        threat_penalty = 0.0
        for sc in my_scs:
            sc_adj = self.get_loc_neighbors(sc)
            threat_count = len(sc_adj & enemy_unit_locs)
            threat_penalty += (threat_count * 20.0)

        return scs_score + scs_pressure_score + mutual_support_score - threat_penalty
    
    def focus_target(self, game_state, mult_target_loc): #refer to docs. focus all troops on 2 provinces
        possible_orders = game_state.get_all_possible_orders()
        orderable_locs = game_state.get_orderable_locations(self.power_name)
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
            # if target_loc in orderable_locs and target_loc not in assigned_units:
            #     hold_order = [o for o in possible_orders[target_loc] if ' H' in o][0]
            #     final_orders.append(hold_order)
            #     assigned_units.add(target_loc)

            #     for loc in orderable_locs:
            #         if loc not in assigned_units:
            #             for order in possible_orders.get(loc, []):
            #                 if ' S ' in order and order.endswith(f' {target_loc}'):
            #                     final_orders.append(order)
            #                     assigned_units.add(loc)
            #                     break

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
                    convoy_attacks = [a for a in attackers if 'VIA' in a[1]]            
                    normal_attacks = [a for a in attackers if 'VIA' not in a[1]]
                    if convoy_attacks: #always prio convoy attacks
                        primary_loc, primary_order = random.choice(convoy_attacks)
                        is_convoy = True
                    else:    
                        primary_loc, primary_order = random.choice(normal_attacks)

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
        for loc in orderable_locs:
            if loc not in assigned_units:
                loc_neighbors = self.get_units_neighbors(loc)
                target1_loc_neighbor = self.get_loc_neighbors(mult_target_loc[0])
                target2_loc_neighbor = self.get_loc_neighbors(mult_target_loc[1])
                scs_unoccupied_neighbors = loc_neighbors & unoccupied_scs
                scs_occupied_neighbors = loc_neighbors & enemy_scs
                target = []
                primary_order = []
                
                if loc_neighbors & common_target:
                    target.append(random.choice(list(loc_neighbors & common_target)))
                    for order in final_orders:
                        if ' S ' not in order and order.endswith(f' - {target[0]}'):
                            primary_order.append(order)
                            break
                elif scs_unoccupied_neighbors:
                    target.append(random.choice(list(scs_unoccupied_neighbors)))
                elif scs_occupied_neighbors:
                    target.append(random.choice(list(scs_occupied_neighbors)))
                elif target1_loc_neighbor & loc_neighbors:
                    target.append(random.choice(list(target1_loc_neighbor & loc_neighbors)))
                elif target2_loc_neighbor & loc_neighbors:
                    target.append(random.choice(list(target2_loc_neighbor & loc_neighbors)))                    

                if target:
                    common_target.add(target[0])
                    if primary_order:
                        for order in possible_orders.get(loc, []):
                            if ' S ' in order and order.endswith(f' {primary_order[0]}'):
                                final_orders.append(order)
                                assigned_units.add(loc)
                                break
                    else:
                        for order in possible_orders.get(loc, []):
                            if self.get_target(order) == target[0]  and ' S ' not in order and ' C ' not in order and 'VIA' not in order:
                                final_orders.append(order)
                                assigned_units.add(loc)
                                break
                else:
                    order = [o for o in possible_orders.get(loc, []) if o.endswith(' H')][0]
                    final_orders.append(order)
        return (final_orders, mult_target_loc)

    def gen_candidate_target(self, game_state, num_targets=20): #generate random targets to focus. currently prioritising attacking and only defending scs
        orderable_locs = game_state.get_orderable_locations(self.power_name)
        if not orderable_locs:
            return []

        possible_orders = game_state.get_all_possible_orders()

        #get all targets avail 
        defense_targets = set(game_state.get_centers(self.power_name)) & set(orderable_locs)
        attack_targets = set()
        for loc in orderable_locs:
            for order in possible_orders.get(loc, []):
                if ' S ' not in order and ' C ' not in order:
                    target = self.get_target(order)
                    if target and target not in defense_targets:
                        attack_targets.add(target)
        reachable_targets = defense_targets | attack_targets

        attack_pool = list(attack_targets)
        defense_pool = list(defense_targets)
        target_pool = list(reachable_targets)
        mult_attack_pool = [list(plan) for plan in itertools.combinations(attack_pool, 2)]
        candidate_targets = self.evaluate_pairs(game_state, mult_attack_pool, num_targets)

        if not target_pool:
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
    def get_action_greedyopp(self, game_state, opp_power):
        self.greedy_opp_agent.game = game_state
        self.greedy_opp_agent.power_name = opp_power
        return self.greedy_opp_agent.get_actions()
        
    
    #base mcts search, depth 1
    def mcts_search(self, game_state, candidate, sims_per_plan):
        if not candidate:
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
                    orders[opp] = self.get_action_greedyopp(sim_game, opp)

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
        #print(final_targets)
        return final_orders
    
    #@timeout_decorator.timeout(1)
    def __init__(self, agent_name='im_horrible_at_giving_nicknames'):
        super().__init__(agent_name)
        self.greedy_opp_agent = GreedyAgent('GreedySim')
        '''Implement your agent here.'''

    #@timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        self.power = game.get_power(self.power_name)

        self.build_map_graphs()

        '''Implement your agent here.'''

    #@timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    #@timeout_decorator.timeout(1)
    def get_actions(self):
        start_time = time.perf_counter()

        '''Implement your agent here.'''
        #non movement phases
        if self.game.phase_type != 'M':
            possible_orders = self.game.get_all_possible_orders()
            my_locs = self.game.get_orderable_locations(self.power_name)
        
            #2D list of choices per location
            # if self.game.phase_type == 'A':
            #     options_per_loc = [[order for order in possible_orders[loc] if order.startswith('A')] for loc in my_locs if possible_orders.get(loc)]
            # else:
            options_per_loc = [possible_orders[loc] for loc in my_locs if possible_orders.get(loc)]
            if not options_per_loc:
                return []

            #get combinations using itertools
            candidate_orders = [(list(plan), "empty") for plan in itertools.product(*options_per_loc)]

            #mcts and evaluate
            #print(time.perf_counter() - start_time, self.game.phase_type)
            return self.mcts_search(self.game, candidate_orders, sims_per_plan=1)
        else:
            candidate_targets = self.gen_candidate_target(self.game)
            candidate_orders = []
            existing_orders = []

            for target in candidate_targets:
                (plan, intended_target) = self.focus_target(self.game, target)
                
                if plan and plan not in existing_orders:
                    existing_orders.append(plan)
                    candidate_orders.append((plan, intended_target))

            #print(time.perf_counter() - start_time, self.game.phase)
            
            return self.mcts_search(self.game, candidate_orders, sims_per_plan=3)

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
