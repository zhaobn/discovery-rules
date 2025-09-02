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
hard_part_rule = reduce(lambda acc,e: f"f_or({e},{acc})", reversed(x[:-1]), x[-1])
hard_rule = f"fand(f_or(pairshape(d,fcircle(),fsquare()),pairshape(d,ftriangle(),fdiamond())),{hard_part_rule})"



ground_truth = medium_rule

# %% define agent
def run_agent(memory = 2, depth = 5, tolerance = 0.9, actions = 10, lastword = 1, inheritance = []):
    
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
    for i in range(actions):

        (program, lp) = guess
        
        # pick a pair that the agent believes will work
        valid_pairs = [ eval(program) == 1 for d in all_pairs ]
        picked_pair_idx = random.choice([i for i,v in enumerate(valid_pairs) if v])
        data_history.append(picked_pair_idx)
        
        # check in the environment
        d = all_pairs[picked_pair_idx]
        is_valid = eval(ground_truth)
        
        if is_valid: # TODO - fix the problem of re-sampling old but good hypothesis
            # stick with this guess, update stats
            programs_history.at[guess_id, "n_success"] += 1
            programs_history.at[guess_id, "n_checks"] += 1

        else:
            # record stats
            programs_history.at[guess_id, "n_checks"] += 1

            # come up with new guess
            is_a_good_guess = False
            while is_a_good_guess is False:
                # sample a new guess
                new_guess = None
                while new_guess is None or 'None' in new_guess[0] or new_guess[0] in programs_history["program"].values:
                    new_guess = agent_prior.generate_tree(logging=False)
                
                # check how good it is at explaining past data
                new_program = new_guess[0]
                history_data = [all_pairs[i] for i in data_history]
                goodness = [ eval(new_program) for d in history_data ]
                
                # save for faster future
                guess_id = len(programs_history)
                programs_history.loc[guess_id] = [new_guess[0], new_guess[1], sum(goodness), len(goodness)]
                
                # accept or keep trying
                if sum(goodness) >= len(goodness) * tolerance:
                    is_a_good_guess = True

            guess = new_guess

    # pass on knowledge - TODO: more control over the choice
    if lastword == 1:
        pass_on = guess[0]

    # performance metrics
    internal_df_length = len(programs_history)
    gt_coverage = [ eval(ground_truth) for d in all_pairs ]
    guess_coverage = [ eval(pass_on) for d in all_pairs ]
    overlap_with_gt = sum(a==b for a,b in zip(gt_coverage, guess_coverage)) / len(all_pairs)

    agent_data = [ all_pairs[i] for i in data_history ]
    guess_itself = [ eval(pass_on) for d in agent_data ]

    return({
        'passed_programs': [ pass_on ],
        'internal_df_length': internal_df_length,
        'guess_overage': sum(guess_coverage) / len(all_pairs),
        'overlap_with_gt': overlap_with_gt,
        'guess_itself': sum(guess_itself) / len(guess_itself)
    })

# %% debug agent
run_agent(actions=40, depth=10, tolerance=0.8)


# %% agent networks
N_GEN = 10

agent_knowledge = run_agent(actions=40, depth=10, tolerance=0.8)
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
