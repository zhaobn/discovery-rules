# %% load packages
import pandas as pd
import random

from functools import reduce

from grammar import *

# %% Set up grammar
productions = [
    ['S', [('A', 0.2), ('B', 0.2), ('fand(S,S)', 0.2), ('f_or(S,S)', 0.3), ('fnot(S)', 0.1), ]],
    ['A', [('C', 0.7), ('sameshape(d)', 0.1), ('isshape(D,X)', 0.1), ('pairshape(d,X,X)', 0.1)]],
    ['B', [('C', 0.7), ('sametexture(d)', 0.1), ('istexture(D,Y)', 0.1), ('pairtexture(d,Y,Y)', 0.1)]],
    ['D', [('obj_1(d)', 0.5), ('obj_2(d)', 0.5)]],
    ['X', [("ftriangle()", 0.25), ("fcircle()", 0.25), ("fsquare()", 0.25), ("fdiamond()", 0.25)]],
    ['Y', [("fplain()", 0.25), ("fcheckered()", 0.25), ("fstripes()", 0.25), ("fdots()", 0.25)]],
    ['C', [('pairshape(d,fcircle(),fsquare())', 1.0)]]
]
test = Mental_grammar(productions, cap=40)
x = test.generate_tree()
#x = ('fand(f_or(pairshape(d,fcircle(),fsquare()),pairshape(d,ftriangle(),fdiamond())),fnot(sametexture(d)))', 0)
print(x)
# d = ((3, 1), (0, 2))
# eval(x[0])

# %% Set up the task
shapes = list(range(4))   # ["triangle", "circle", "square", "diamond"]
textures = list(range(4)) # ["plain", "checkered", "stripes", "dots"]

all_objs = [(s, t) for s in shapes for t in textures]
all_pairs = [(m, n) for m in all_objs for n in all_objs if m != n]

easy_rule = 'sameshape(d)'
medium_rule = 'fand(f_or(pairshape(d,fcircle(),fsquare()),pairshape(d,ftriangle(),fdiamond())),fnot(sametexture(d)))'

hard_part = [
    'fand(istexture(obj_1(d),fdots()),istexture(obj_2(d),fdots()))',
    'fand(istexture(obj_1(d),fdots()),istexture(obj_2(d),fstripes()))',
    'fand(istexture(obj_1(d),fdots()),istexture(obj_2(d),fcheckered()))',
    'fand(istexture(obj_1(d),fdots()),istexture(obj_2(d),fplain()))',
    'fand(istexture(obj_1(d),fstripes()),istexture(obj_2(d),fstripes()))',
    'fand(istexture(obj_1(d),fstripes()),istexture(obj_2(d),fcheckered()))',
    'fand(istexture(obj_1(d),fstripes()),istexture(obj_2(d),fplain()))',
    'fand(istexture(obj_1(d),fcheckered()),istexture(obj_2(d),fcheckered()))',
    'fand(istexture(obj_1(d),fcheckered()),istexture(obj_2(d),fplain()))',
    'fand(istexture(obj_1(d),fplain()),istexture(obj_2(d),fplain()))',
]
hard_part_rule = reduce(lambda acc,e: f"f_or({e},{acc})", reversed(hard_part[:-1]), hard_part[-1])
hard_rule = f"fand(f_or(pairshape(d,fcircle(),fsquare()),pairshape(d,ftriangle(),fdiamond())),{hard_part_rule})"


# %% define agent
def run_agent(
        actions=10, cap=10, depth=5, env_rule=easy_rule, inheritance=[],
        lastword=1, memory=2, tolerance=0.9    
    ):
    
    # set up agent 
    if len(inheritance) > memory:
        inherited = random.sample(inheritance, memory)
    else:
        inherited = inheritance
    
    agent_lib = productions.copy()
    if len(inheritance) > 0:
        agent_lib[-1] = ['C', inherited]

    agent_prior= Mental_grammar(productions, cap=depth)

    # internal records
    programs_history = pd.DataFrame(columns=["program","prior","n_success","n_checks"])
    data_history = []

    # sample a guess
    guess = None
    while guess is None or 'None' in guess[0]:
        guess = agent_prior.generate_tree(logging=False)
    
    guess_id = len(programs_history)
    programs_history.loc[guess_id] = [guess[0], guess[1], 0, 0]

    # main stuff
    n_hit = []
    for i in range(actions):

        (program, lp) = guess
        
        # pick a pair that the agent believes will work
        valid_pairs = [ eval(program) == 1 for d in all_pairs ]
        if sum(valid_pairs) > 0:
            picked_pair_idx = random.choice([i for i,v in enumerate(valid_pairs) if v])
        else:
            picked_pair_idx = random.choice(list(range(len(all_pairs))))
        data_history.append(picked_pair_idx)
        
        # check in the environment
        d = all_pairs[picked_pair_idx]
        is_valid = eval(env_rule)
        
        if is_valid:
            n_hit.append(1)

            # stick with this guess, update stats
            programs_history.at[guess_id, "n_success"] += 1
            programs_history.at[guess_id, "n_checks"] += 1

        else:
            n_hit.append(0)

            # come up with another guesses
            is_a_good_guess = False
            n_loop = 0
            
            while is_a_good_guess is False and n_loop <= cap:

                n_loop += 1 
                
                # sample a new guess
                new_guess = None
                while new_guess is None or 'None' in new_guess[0]:
                    new_guess = agent_prior.generate_tree(logging=False)

                new_program = new_guess[0]
                if new_program in programs_history["program"].values:
                    guess_id = programs_history.index[programs_history["program"] == new_program].tolist()[0]
                else:
                    guess_id = len(programs_history)
                
                # check how good it is at explaining past data
                data_list = [all_pairs[i] for i in data_history]
                goodness = [ eval(new_program) == eval(env_rule) for d in data_list ]
                programs_history.loc[guess_id] = [new_guess[0], new_guess[1], sum(goodness), len(goodness)]
                
                # accept or keep trying
                if sum(goodness) >= len(goodness) * tolerance:
                    is_a_good_guess = True
            
            if is_a_good_guess:
                guess = new_guess


    # performance metrics
    (current_guess, lp) = guess
    gt_coverage = [ eval(env_rule) for d in all_pairs ]
    guess_coverage = [ eval(current_guess) for d in all_pairs ]
    overlap_with_gt = sum(a==b for a,b in zip(gt_coverage, guess_coverage)) / len(all_pairs)

    # pass on knowledge
    data_list = [ all_pairs[i] for i in data_history ]
    for i, program in programs_history["program"].items():
        perf = [ eval(program) == eval(env_rule) for d in data_list ]
        programs_history.at[i, "n_success"] = sum(perf)
        programs_history.at[i, "n_checks"] = len(perf)
    
    filtered = programs_history[programs_history["n_success"] / programs_history["n_checks"] >= tolerance ]
    if len(filtered) >= lastword:
        log_weights = np.log(filtered["n_success"]) #+ filtered["prior"]
        weights = np.exp(log_weights - log_weights.max())  # stabilize
        sampled = filtered.sample(lastword, weights=weights)

    elif len(programs_history) > lastword:
        log_weights = np.log(programs_history["n_success"]) #+ programs_history["prior"]
        weights = np.exp(log_weights - log_weights.max())
        sampled = programs_history.sample(lastword, weights=weights)
    
    else:
        sampled = programs_history

    pass_on = sampled['program'].tolist()
    
    overlap_with_msg = [0]
    if len(inheritance) > 0:
        overlap_with_msg = []
        for new_msg in pass_on:
            new_msg_coverage = [ eval(new_msg) for d in all_pairs ]
            
            for msg in inheritance:
                msg_coverage = [ eval(msg) for d in all_pairs ]
                msg_overlap = sum(a==b for a,b in zip(new_msg_coverage, msg_coverage)) / len(all_pairs)

                overlap_with_msg.append(msg_overlap)

    return({
        #'last_guess': current_guess,
        'passed_programs': pass_on,
        'internal_df_length': len(programs_history),
        'overlap_with_gt': overlap_with_gt,
        'agent_success': sum(n_hit)/len(n_hit),
        'agent_change': max(overlap_with_msg)
    })

# # %% debug agent
# run_agent(actions=10, depth=10, tolerance=0.95, lastword=1, env_rule=hard_rule)

# %% agent networks
N_CHAIN = 1
N_GEN = 10
RULES = {
    'easy': easy_rule,
    'med': medium_rule,
    'hard': hard_rule
}

sim_stats = pd.DataFrame(columns=["rule_type","chain_id","gen_id", "success","gt_overlap","msg_overlap","internal_length"])
sim_record = pd.DataFrame(columns=["rule_type","chain_id","gen_id", "msg_id","program"]) # - TODO: learn from multiple ancesters

for rule_type, rule_content in RULES.items():
    for chain in range(N_CHAIN):    
        messages = []

        for gen in range(N_GEN):
            print(f"running the {rule_type} rule for chain {chain}, generation {gen}...")
            result = run_agent(
                inheritance=messages, 
                env_rule=rule_content,
                cap=100, depth=6, lastword=2
            )

            # update stats
            sim_stats.loc[len(sim_stats)] = [
                rule_type,
                chain, gen, result['agent_success'], result['overlap_with_gt'],
                result['agent_change'], result['internal_df_length']
            ]

            # update messages record
            for msg_id, program in enumerate(result['passed_programs']):
                sim_record.loc[len(sim_record)] = [rule_type, chain, gen, msg_id, program]

            messages = result['passed_programs']
            print(f"program passed: {messages}")

# %%