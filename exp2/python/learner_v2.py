# %% 
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Set up task and environment
colors = np.arange(6)
shapes = np.arange(4)   # ["triangle", "circle", "square", "diamond"]
textures = np.arange(4) # ["plain", "checkered", "stripes", "dots"]

norm_objs = [(s, t) for s in shapes for t in textures]
state_fmt = np.zeros((len(norm_objs), len(colors)))
obj_to_idx = {obj: i for i, obj in enumerate(norm_objs)}

norm_pairs = [(m, n) for m in norm_objs for n in norm_objs if m != n]

def simple_task (pair):
    (m, n) = pair
    return m[0] == n[0]

def med_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] != n[1]

def test_rule (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and ( m[1] % 2 == 0 or n[1] % 2 == 0 ) 

def task_func(task, pair):
    if task == 'simple':
        return simple_task(pair)
    elif task == 'med':           
        return med_task(pair)
    elif task == 'hard':
        return test_rule(pair)
    else:
        print('no matching condition!')
        return None

# %%
def update_state(condition, state, action):
    reward = 0
    signal = 'S' # success

    if (len(action) == 3):
        shape, texture, level = action
        state[obj_to_idx[(shape, texture)], level] -= 1
        reward = 10 ** level
    
    elif (len(action) == 2):
        obj1, obj2 = action
        pair = ((obj1[0], obj1[1]), (obj2[0], obj2[1])) 
        is_valid = task_func(condition, pair) 

        if is_valid:
            new_level = min(max(colors), max(obj1[2], obj2[2]) + 1)
            state[obj_to_idx[(obj1[0], obj2[1])], new_level] += 1

            state[obj_to_idx[(obj1[0], obj1[1])], obj1[2]] -= 1
            state[obj_to_idx[(obj2[0], obj2[1])], obj2[2]] -= 1
        
        else:
            signal = 'F'

    else:
        raise ValueError("Wrong action format")

    return (state, reward, signal)

# %%
def policy(step, horizon, current_state, belief):
    # find the most valuable object
    cur_max_level = np.max(np.nonzero(current_state)[1])
    objs_at_max = [norm_objs[i] for i in np.nonzero(current_state[:, cur_max_level])[0]]

    # if improvable, improve


    # otherwise consume



# %%
def run_condition(condition, num_episodes=100, num_actions=40, seed=0):
    print(f"Running base condition: {condition}, with seed {seed}")
    np.random.seed(seed)

    # Initialize stats variables
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    epi_probs = np.zeros(num_episodes)
    recover_rate = np.zeros(num_episodes)

    # Initialize prior
    prior_alphas = np.full(len(norm_pairs), 0.001)
    prior_betas = np.full(len(norm_pairs), 0.001)

    # Prep for ground truth measure
    true_transitions = np.array([task_func(condition, pair) for pair in norm_pairs]).astype(int)

    for episode in range(num_episodes):

        # Initialize state
        state = state_fmt.copy()
        state[:, 0] = 1
        reward = 0
        max_level = 0

        # Prep for iteration
        sampled_probs = np.random.beta(prior_alphas, prior_betas)
        sampled_transitions = np.random.binomial(1, sampled_probs)
        epi_probs[episode] = sum(sampled_transitions)/len(sampled_transitions)
        recover_rate[episode] = (true_transitions == sampled_transitions).sum() / len(sampled_transitions)
        
        for t in range(num_actions):
            action = policy(t, num_actions, state, sampled_transitions)

            (new_state, immediate_reward, signal) = update_state(condition, state, action)
            
            state = new_state
            reward += immediate_reward
            cum_rewards[episode, t] = max(cum_rewards[episode, t-1], reward)

            max_level = np.max(np.where(state > 0)[1])
            highst_levels[episode, t] = max_level

            if len(action) == 2:
                obj1, obj2 = action
                pair_idx = norm_pairs.index(((obj1[0], obj1[1]), (obj2[0], obj2[1])))

                if signal == 'F':
                    prior_alphas[pair_idx] += 1
                else:
                    prior_betas[pair_idx] += 1
                    
    return condition, cum_rewards, highst_levels, epi_probs, recover_rate


# %%
# Claude code to be studied
def agent_policy(state, belief, norm_objs, norm_pairs, max_level_cap):
    # Step 1: Find all objects at highest level
    max_level = np.max(np.nonzero(state)[1])
    objs_at_max_indices = np.nonzero(state[:, max_level])[0]
    
    if len(objs_at_max_indices) == 0:
        return None  # No objects available
    
    # Step 2: If not at cap, find best combinable object
    if max_level < max_level_cap:
        # Build lookup for which objects are available at which levels (do once)
        obj_availability = {}
        for obj_idx, obj in enumerate(norm_objs):
            available_levels = np.nonzero(state[obj_idx, :])[0]
            if len(available_levels) > 0:
                obj_availability[obj] = available_levels.min()
        
        # Build lookup for pairs by first/second object (do once)
        pairs_by_obj = {}
        for pair_idx, pair in enumerate(norm_pairs):
            if belief[pair_idx] > 0:  # Only consider believed transitions
                if pair[0] not in pairs_by_obj:
                    pairs_by_obj[pair[0]] = []
                if pair[1] not in pairs_by_obj:
                    pairs_by_obj[pair[1]] = []
                pairs_by_obj[pair[0]].append((pair[1], pair_idx))
                pairs_by_obj[pair[1]].append((pair[0], pair_idx))
        
        # Check each max-level object for best combine
        best_obj = None
        best_combine = None
        lowest_partner_level = float('inf')
        
        for obj_idx in objs_at_max_indices:
            obj = norm_objs[obj_idx]
            
            # Skip if this object has no valid pairs
            if obj not in pairs_by_obj:
                continue
            
            # Check all partners for this object
            for partner, pair_idx in pairs_by_obj[obj]:
                if partner in obj_availability:
                    partner_level = obj_availability[partner]
                    
                    if partner_level < lowest_partner_level:
                        lowest_partner_level = partner_level
                        best_obj = obj
                        best_combine = (obj, partner, max_level, partner_level)
        
        # If we found a valid combine, return it
        if best_combine is not None:
            obj1, obj2, level1, level2 = best_combine
            return ('combine', (obj1[0], obj1[1], level1), (obj2[0], obj2[1], level2))
    
    # Step 3: No valid combines found, consume first max-level object
    best_obj = norm_objs[objs_at_max_indices[0]]
    return ('consume', (best_obj[0], best_obj[1], max_level))