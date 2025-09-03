# %% load packages
import pandas as pd
import random

from functools import reduce

from grammar import *

# %% Set up grammar
productions = [
  ['S', ['A', 'B', 'fand(S,S)', 'fnot(S)', 'f_or(S,S)']],
  ['A', ['C', 'sameshape(d)', 'isshape(D,X)', 'pairshape(d,X,X)']],
  ['B', ['C', 'sametexture(d)', 'istexture(D,Y)', 'pairtexture(d,Y,Y)']],
  ['D', ['obj_1(d)', 'obj_2(d)']],
  ['X', ["ftriangle()", "fcircle()", "fsquare()", "fdiamond()"]],
  ['Y', ["fplain()", "fcheckered()", "fstripes()", "fdots()"]],
  ['C', ['None']]
]
# test = Mental_grammar(productions, cap=100)
# x = test.generate_tree()
# x = ('fand(f_or(pairshape(d,fcircle(),fsquare()),pairshape(d,ftriangle(),fdiamond())),fnot(sametexture(d)))', 0)
# print(x)
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



ground_truth = medium_rule

# %% define agent
def run_agent(memory=2, depth=5, tolerance=0.9, actions=10, cap=10, lastword=1, inheritance=[]):
    
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
        is_valid = eval(ground_truth)
        
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
            
            while is_a_good_guess is False or n_loop <= cap:

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
                goodness = [ eval(new_program) == eval(ground_truth) for d in data_list ]
                programs_history.loc[guess_id] = [new_guess[0], new_guess[1], sum(goodness), len(goodness)]
                
                # accept or keep trying
                if sum(goodness) >= len(goodness) * tolerance:
                    is_a_good_guess = True
            
            if is_a_good_guess:
                guess = new_guess


    # performance metrics
    (current_guess, lp) = guess
    gt_coverage = [ eval(ground_truth) for d in all_pairs ]
    guess_coverage = [ eval(current_guess) for d in all_pairs ]
    overlap_with_gt = sum(a==b for a,b in zip(gt_coverage, guess_coverage)) / len(all_pairs)

    internal_df_length = len(programs_history)

    # pass on knowledge
    data_list = [ all_pairs[i] for i in data_history ]
    for i, program in programs_history["program"].items():
        perf = [ eval(program) == eval(ground_truth) for d in data_list ]
        programs_history.at[i, "n_success"] = sum(perf)
        programs_history.at[i, "n_checks"] = len(perf)
    
    filtered = programs_history[programs_history["n_success"] / programs_history["n_checks"] >= tolerance ]
    if len(filtered) >= lastword:
        log_weights = np.log(filtered["n_success"]) + filtered["prior"]
        weights = np.exp(log_weights - log_weights.max())  # stabilize
        sampled = filtered.sample(lastword, weights=weights)

    elif len(programs_history) >= lastword:
        log_weights = np.log(programs_history["n_success"]) + programs_history["prior"]
        weights = np.exp(log_weights - log_weights.max())
        sampled = programs_history.sample(lastword, weights=weights)
    
    else:
        sampled = programs_history

    pass_on = sampled['program'].tolist()
   

    return({
        'last_guess': current_guess,
        'passed_programs': pass_on,
        'internal_df_length': internal_df_length,
        'guess_overage': sum(guess_coverage) / len(all_pairs),
        'overlap_with_gt': overlap_with_gt,
        #'agent_history': n_hit,
    })

# %% debug agent
ground_truth = easy_rule
run_agent(actions=10, depth=10, tolerance=0.95, lastword=2)



# %% agent networks
N_GEN = 10
N_CHAIN = 5

coverage_history = []
liblen_history = []



agent_knowledge = run_agent(actions=20, depth=10, tolerance=0.9)
# TODO: learn from many ancesters?
for k in range(N_GEN):
    agent_knowledge = run_agent(actions=40, depth=10, tolerance=0.8, inheritance=agent_knowledge['passed_programs'])


# # %% run simulations
# N_GEN = 10
# N_AGENTS = 3

# overlap_metrics = []
# lengths_metrics = []
# diversity_metrics = []

# messages = []
# for i in range(N_GEN):

#     gen_messages = []
#     for a in N_AGENTS:
#         output = run_agent(inheritance = messages)
#         messages.append(output[0])
#         gen_messages.append(output[0])

#         # compute performance metrics
    
#     messages = gen_messages

# %%
