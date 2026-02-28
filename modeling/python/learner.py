# %% 
# Load packages
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from grammar import *

# %%
# Set up task and environment
colors = np.arange(6)
max_level_cap = max(colors)

shapes = np.arange(4)   # ["triangle", "circle", "square", "diamond"]
textures = np.arange(4) # ["plain", "checkered", "stripes", "dots"]

norm_objs = [(s, t) for s in shapes for t in textures]
state_fmt = np.zeros((len(norm_objs), len(colors)))
obj_to_idx = {obj: i for i, obj in enumerate(norm_objs)}

norm_pairs = [(m, n) for m in norm_objs for n in norm_objs if m != n]
pair_to_idx = {pair: i for i, pair in enumerate(norm_pairs)}

def simple_task (pair):
    (m, n) = pair
    return m[0] == n[0]

def med_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] != n[1]

def hard_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] >= n[1]

def test_rule (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and ( m[1] % 2 == 0 or n[1] % 2 == 0 ) 

def test_rule_2 (pair):
    (m, n) = pair
    return m[1] + n[1] == 3 and ( m[0] % 2 == 0 or n[0] % 2 == 0 ) 

def test_rule_3 (pair):
    (m, n) = pair
    return m[1] + n[1] == 3 and ( m[0] % 2 == 1 or n[0] % 2 == 0 ) 



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
# check rule coverage
get_coverage = lambda task: sum([task(pair) for pair in norm_pairs])

get_coverage(simple_task)   #48
get_coverage(med_task)      #48
get_coverage(hard_task)     #40
get_coverage(test_rule)     #48
get_coverage(test_rule_2)   #48
get_coverage(test_rule_3)   #48

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
            new_level = min(max_level_cap, max(obj1[2], obj2[2]) + 1)
            state[obj_to_idx[(obj1[0], obj2[1])], new_level] += 1

            state[obj_to_idx[(obj1[0], obj1[1])], obj1[2]] -= 1
            state[obj_to_idx[(obj2[0], obj2[1])], obj2[2]] -= 1
        
        else:
            signal = 'F'

    else:
        raise ValueError("Wrong action format")

    return (state, reward, signal)

# %%
def sample_obj(index_list):
    obj_idx, level = random.choice(index_list)
    shape, texture = norm_objs[obj_idx]
    return (shape, texture, level)

def local_greedy_policy(state, sampled_transitions, n_left):
    
    # --- find available objects ---
    available = np.argwhere(state > 0)
    if len(available) == 0:
        return None  # no actions possible
    
    # --- find current highest level ---
    highest_level = np.max(available[:, 1])
    top_objs = available[available[:, 1] == highest_level]

    # --- if one action left or hit the max level, consume ---
    if n_left == 1 or highest_level >= max_level_cap:
        chosen_obj = sample_obj(top_objs)
        return chosen_obj
    
    # --- attempt to find a valid combination to improve top-level items ---
    partners = sorted([tuple(p) for p in np.argwhere(state > 0)], key=lambda x: x[1])  # sort once globally
    
    np.random.shuffle(top_objs) # Try each top-level object in random order for exploration
    for top_idx, level_top in top_objs:
        obj_top = norm_objs[top_idx]

        # iterate through partners, lowest level first
        for partner_idx, partner_level in partners:
            # skip pairs with identical objs
            if (top_idx == partner_idx) and (level_top == partner_level):
                continue
            
            obj_partner = norm_objs[partner_idx]
            
            pair = (obj_top, obj_partner)
            pair_index = pair_to_idx.get(pair, None)
            if pair_index is not None and sampled_transitions[pair_index] == 1:
                # Found valid combination
                return (
                    (obj_top[0], obj_top[1], level_top),
                    (obj_partner[0], obj_partner[1], partner_level)
                )
            
    # no valid combination, harvest highest-level object
    chosen_obj = sample_obj(top_objs)
    return chosen_obj


# %%
pcfg_prior = pd.read_csv('models_v1/tree_mdps.csv')
pair_columns = [f'pair_{i}' for i in range(240)]

def run_condition(model='base', condition='simple', num_episodes=100, num_actions=40, seed=0):
    print(f"Running {model} model for condition: {condition}, with seed {seed}")
    np.random.seed(seed)

    # Initialize stats
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    epi_probs = np.zeros(num_episodes)
    recover_rate = np.zeros(num_episodes)

    # Initialize prior
    if model == 'pcfg':
        prior_mdps = pcfg_prior.copy()
    else:
        prior_alphas = np.full(len(norm_pairs), 0.001)
        prior_betas = np.full(len(norm_pairs), 0.001)

    # Prep for ground truth measure
    true_transitions = np.array([task_func(condition, pair) for pair in norm_pairs]).astype(int)

    for episode in range(num_episodes):

        # Reset environment
        state = state_fmt.copy()
        state[:, 0] = 2
        reward = 0
        max_level = 0

        # Prep for iteration
        if model == 'pcfg':
            if len(prior_mdps) == 0:
                print("No valid prior MDPs left.")
                break
            else:
                counts = prior_mdps['count']
                norm_probs = counts / counts.sum()
                scaled_probs = norm_probs
                prior_mdp_index = np.random.choice(prior_mdps.index, size=1, p=scaled_probs)[0]
                sampled_transitions = prior_mdps.loc[prior_mdp_index, pair_columns].to_numpy(dtype=int)
        
        else:
            sampled_probs = np.random.beta(prior_alphas, prior_betas)
            sampled_transitions = np.random.binomial(1, sampled_probs)
        
        epi_probs[episode] = sampled_transitions.mean()
        recover_rate[episode] = (true_transitions == sampled_transitions).mean()
        
        for t in range(num_actions):
            action = local_greedy_policy(state, sampled_transitions, num_actions-t)

            if action is None:
                cum_rewards[episode, t:] = reward
                highst_levels[episode, t:] = max_level
                break

            new_state, immediate_reward, signal = update_state(condition, state, action)
            
            state = new_state
            reward += immediate_reward
            cum_rewards[episode, t] = reward # max(cum_rewards[episode, t-1], reward)

            max_level = np.max(np.where(state > 0)[1], initial=max_level)
            highst_levels[episode, t] = max_level

            if len(action) == 2: # combine
                obj1, obj2 = action
                pair_idx = norm_pairs.index(((obj1[0], obj1[1]), (obj2[0], obj2[1])))

                if model == 'pcfg':
                    pair_column = f'pair_{pair_idx}'
                    if signal == 'F':
                        prior_mdps = prior_mdps[prior_mdps[pair_column] == 0]
                    else:
                    # with some probability, remove inconsistent entries
                        if np.random.rand() < 0.02:
                            prior_mdps = prior_mdps[prior_mdps[pair_column] == 1]

                else:
                    if signal == 'S':
                        prior_alphas[pair_idx] += 1
                    else:
                        prior_betas[pair_idx] += 1
                    
    log_cum_rewards = np.log(cum_rewards + 1e-8)  # add small epsilon to avoid log(0)
    mean_log_cum_rewards = log_cum_rewards.mean(axis=0) # shape: (num_actions,)
    mean_highest_levels = highst_levels.mean(axis=0)

    return condition, cum_rewards, highst_levels, mean_log_cum_rewards, mean_highest_levels, epi_probs, recover_rate


# %%
run_condition('pcfg', 'med', 10, 10, 0)

# %%
episode = 100
actions = 40
n_seeds = 10
save_path = 'simulation_results/base_psrl_sim.pkl'

results = []
for condition in ['simple', 'med', 'hard']:
    for seed in range(10):
        condition_name, cum_rewards, highst_levels, mean_log_cum_rewards, mean_highest_levels, epi_probs, recover_rate = run_condition('base', condition, episode, actions, seed)            
        
        # Create one row per action (step)
        for action_idx in range(actions):
            results.append({
                'condition': condition_name,
                'seed': seed,
                'action': action_idx,
                'mean_log_cum_rewards': mean_log_cum_rewards[action_idx],
                'mean_highest_levels': mean_highest_levels[action_idx],
                'epi_probs': epi_probs[action_idx],
                'recover_rate': recover_rate[action_idx],
            })

df = pd.DataFrame(results)
df.to_pickle(save_path)


# %%
episode = 100
actions = 40
n_seeds = 20
save_path = 'simulation_results/pcfg_psrl_sim_4.pkl'

results = []
for condition in ['simple', 'med', 'hard']:
    for seed in range(n_seeds):
        (condition_name, cum_rewards, highst_levels, mean_log_cum_rewards, 
         mean_highest_levels, epi_probs, recover_rate) = run_condition(
            'pcfg', condition, episode, actions, seed)        
        # Create one row per action (step)
        for action_idx in range(actions):
            results.append({
                'condition': condition_name,
                'seed': seed,
                'action': action_idx,
                'mean_log_cum_rewards': mean_log_cum_rewards[action_idx],
                'mean_highest_levels': mean_highest_levels[action_idx],
                'epi_probs': epi_probs[action_idx],
                'recover_rate': recover_rate[action_idx],
            })

df = pd.DataFrame(results)
df.to_pickle(save_path)

# %%
# save_path = 'simulation_results/base_psrl_sim_cogsci.pkl'
# save_path = 'simulation_results/base_psrl_sim_new.pkl'
save_path = 'simulation_results/pcfg_psrl_sim_4.pkl'

df = pd.read_pickle(save_path)
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['mean_log_cum_rewards', 'mean_highest_levels', 'epi_probs', 'recover_rate']
titles = ['Mean Cumulative Rewards (log)', 'Mean Highest Levels', 'Episode Probabilities', 'Recovery Rate']

for ax, metric, title in zip(axes.flat, metrics, titles):
    sns.lineplot(data=df, x='action', y=metric, hue='condition', ax=ax, errorbar='sd')
    ax.set_title(title)
    ax.set_xlabel('Action Step')
    
plt.tight_layout()
plt.show()

# %%
