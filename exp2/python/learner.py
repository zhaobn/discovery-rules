# %% 
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import traceback

# %%
# Set up task and environment
colors = np.arange(6)
shapes = np.arange(4)   # ["triangle", "circle", "square", "diamond"]
textures = np.arange(4) # ["plain", "checkered", "stripes", "dots"]

norm_objs = [(s, t) for s in shapes for t in textures]
norm_pairs = [(m, n) for m in norm_objs for n in norm_objs if m != n]

def simple_task (pair):
    (m, n) = pair
    return m[0] == n[0]

def med_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] != n[1]

def hard_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] >= n[1]

def task_func(task, pair):
    if task == 'simple':
        return simple_task(pair)
    elif task == 'med':
        return med_task(pair)
    elif task == 'hard':
        return hard_task(pair)

# %%
# Base PSRL agent
def find_combination(state_objs, state_levels, sampled_transitions):
    state_objs = np.array(state_objs)

    # Create a mapping from level to object indices, sorted by level (highest to lowest)
    level_to_objects = {}
    for i, level in enumerate(state_levels):
        # Only include non-negative levels
        if level >= 0:
            if level not in level_to_objects:
                level_to_objects[level] = []
            level_to_objects[level].append(i)
    sorted_levels = sorted(level_to_objects.keys(), reverse=True)

    # Check each level from highest to lowest
    for level in sorted_levels:
        # Get all objects at this level
        obj_indices = level_to_objects[level].copy()

        # Continue until all objects at this level have been checked
        while obj_indices:
            # Randomly sample the next object to check
            random_idx = np.random.choice(len(obj_indices))
            obj_idx = obj_indices.pop(random_idx)
            obj = state_objs[obj_idx]

            # Find pairs where this object can be combined (transitions == 1)
            matched_pairs = [pair for pair in norm_pairs
                if sampled_transitions[norm_pairs.index(pair)] == 1
                and any(np.array_equal(p, obj) for p in pair)
            ]

            if matched_pairs:
                matched_objs = [pair[0] if np.array_equal(pair[1], obj) else pair[1] for pair in matched_pairs]
                intersection = []
                for matched_obj in matched_objs:
                    for i, state_obj in enumerate(state_objs):
                        if np.array_equal(matched_obj, state_obj):
                            intersection.append(i)         

                if intersection:
                    potential_partners = []
                    for idx in intersection:
                        # Compare face values (the first and second elements in the tuple)
                        if (state_objs[idx][0] != obj[0] or state_objs[idx][1] != obj[1]):
                            potential_partners.append(idx)

                    if potential_partners:
                        # Randomly choose a partner with different face value
                        chosen_idx = np.random.choice(len(potential_partners))
                        partner_idx = potential_partners[chosen_idx]
                    
                    return [(state_objs[obj_idx][0], state_objs[obj_idx][1], state_levels[obj_idx]), 
                            (state_objs[partner_idx][0], state_objs[partner_idx][1], state_levels[partner_idx])]
       
                    
        
    # No valid combination found
    return None

def policy(actions_left, state_objs, state_levels, sampled_transitions):

    if len(state_objs) == 0:
        return None

    highest_level = np.max(state_levels)
    highest_level_obj_idx = np.where(state_levels == highest_level)[0]
    

    if actions_left < 2 or highest_level == len(colors):
        selected_obj = state_objs[np.random.choice(highest_level_obj_idx)]
        return [ (selected_obj[0], selected_obj[1], highest_level) ]
    
    else:
        action = find_combination(state_objs, state_levels, sampled_transitions)
        if action is not None:
            return action
        
        else:
            # check if there is any item left
            left_items = np.where(state_levels > -1)[0]
            if left_items.size:
                chosen_obj_idx = np.random.choice(left_items)
                return [(state_objs[chosen_obj_idx][0], state_objs[chosen_obj_idx][1], state_levels[chosen_obj_idx])]
            else:
                return None


def get_reward(action):
    if action is None or len(action) == 2:
        return 0
    else:
        return 10 ** action[0][2]

def update_states(task, action, action_left, state_objs, state_levels, regeneratable):

    action_left = action_left - 1
    if action is None or action_left < 0:
        return None

    state_objs = np.array(state_objs)
    state_levels = np.array(state_levels)
    regeneratable = np.array(regeneratable)

    new_state_objs = state_objs
    new_state_levels = state_levels
    new_regeneratable = regeneratable

    if len(action) == 2:
        [m, n] = action
        norm_m = (m[0], m[1])
        norm_n = (n[0], n[1])
        
        is_valid = task_func(task, (norm_m, norm_n))
        
        if is_valid:
            new_obj = (norm_m[0], norm_n[1])
            new_level = np.max([m[2], n[2]]) + 1

            # check if remove or keep the used objects
            for item in [norm_m, norm_n]:
                m_index = np.where(np.all(state_objs == item, axis=1))[0][0]
                #m_index = state_objs.index(item)
                if state_levels[m_index] == 0 and regeneratable[m_index] == 1:
                    new_regeneratable[m_index] = 0

                else:
                    new_state_objs = np.delete(state_objs, [m_index], axis=0)
                    new_state_levels = np.delete(state_levels, [m_index], axis=0)
                    new_regeneratable = np.delete(regeneratable, [m_index], axis=0)

            # Add new_obj and new_level to state_objs and state_levels
            new_state_objs = np.append(new_state_objs, [new_obj], axis=0)
            new_state_levels = np.append(new_state_levels, new_level)
            new_regeneratable = np.append(regeneratable, 0)
    else:
        (m, n, l) = action[0]
        item = (m, n)
        item_idx = np.where(np.all(state_objs == item, axis=1))[0][0]
        #item_idx = state_objs.index(item)
        if state_levels[item_idx] == 0 and regeneratable[item_idx] == 1:
            new_regeneratable[item_idx] = 0
        
        else:
            new_state_objs = np.delete(state_objs, [item_idx], axis=0)
            new_state_levels = np.delete(state_levels, [item_idx], axis=0)
            new_regeneratable = np.delete(regeneratable, [item_idx], axis=0)

    return (action_left, new_state_objs, new_state_levels, new_regeneratable)



# %%
# load prior beliefs
def softmax(x, tau=1.0):
    x = np.array(x) / tau  # Scale by temperature
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def run_pcfg_condition(condition, num_episodes=10, num_actions=40):
    print(f"Running pcfg condition: {condition}")

    # Initialize variables
    task = condition
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    # actions = np.full((num_episodes, num_actions), None)
    epi_probs = np.zeros(num_episodes)
    recover_rate = np.zeros(num_episodes)

    # Initialize the prior
    prior_mdps = pd.read_csv('tree_mdps.csv')
    pair_columns = [f'pair_{i}' for i in range(240)]
    
    # Prep for ground truth measure
    true_transitions =  np.array([task_func(task, pair) for pair in norm_pairs]).astype(int)

    # Run PCFG PSRL agent
    for episode in range(num_episodes):

        if len(prior_mdps) == 0:
            print("No valid prior MDPs left.")
            break

        # Sample from the prior
        counts = prior_mdps['count']
        norm_probs = counts / counts.sum()
        scaled_probs = norm_probs #softmax(norm_probs, tau=0.01)

        prior_mdp_index = np.random.choice(prior_mdps.index, size=1, p=scaled_probs)[0]
        sampled_transitions = prior_mdps.loc[prior_mdp_index][pair_columns].to_numpy()

        # # Compute stats
        # epi_probs[episode] = sum(sampled_transitions)/len(sampled_transitions)
        recover_rate[episode] = (true_transitions == sampled_transitions).sum() / len(sampled_transitions)

        # Compute expected loss
        all_transitions = prior_mdps[pair_columns].to_numpy()
        mdp_probs = np.sum(all_transitions, axis=1) / all_transitions.shape[1]
        recovery_matches = (all_transitions == true_transitions).sum(axis=1) / all_transitions.shape[1]
    
        weights = prior_mdps['count'].to_numpy()
        weights = weights / weights.sum()
    
        expected_prob = np.sum(mdp_probs * weights)
        expected_recovery_rate = np.sum(recovery_matches * weights)

        epi_probs[episode] = expected_prob
        recover_rate[episode] = expected_recovery_rate

        # Prepare for the episode
        state_objs = norm_objs.copy()
        state_levels = np.zeros(len(norm_objs))

        reward = 0
        regeneratable = np.ones(len(norm_objs))

        max_level = 0
        for t in range(num_actions):
            action = policy(num_actions - t, state_objs, state_levels, sampled_transitions)
            if t > 0:
                cum_rewards[episode, t] = cum_rewards[episode, t-1] + get_reward(action)
            
            if action is None:
                if t > 0:
                    highst_levels[episode, t] = highst_levels[episode, t-1]
            
            else:

                (_, state_objs, state_levels, regeneratable) = update_states(task, action, num_actions - t, state_objs, state_levels, regeneratable)

                if len(state_objs) > 0:
                    new_highest = np.max(state_levels)
                    max_level = max(max_level, new_highest)
                
                highst_levels[episode, t] = max_level

                if len(action) == 2:
                    [m, n] = action
                    norm_m = (m[0], m[1])
                    norm_n = (n[0], n[1])

                    pair_idx = norm_pairs.index((norm_m, norm_n))
                    pair_column = f'pair_{pair_idx}'

                    # remove inconsistent entries from prior_mdps
                    if task_func(task, (norm_m, norm_n)) == False:
                        prior_mdps = prior_mdps[prior_mdps[pair_column] == 0]
                    # else:
                    #     # with some probability, remove inconsistent entries
                    #     if np.random.rand() < 0.1:
                    #         prior_mdps = prior_mdps[prior_mdps[pair_column] == 1]
                        
                    if len(prior_mdps) == 0:
                        print("No valid prior MDPs left.")
                        break

                    # if task_func(task, (norm_m, norm_n)):
                    #     prior_mdps = prior_mdps[prior_mdps[pair_column] == 1]
                    
                    # else:
                    #     prior_mdps = prior_mdps[prior_mdps[pair_column] == 0]

                    

    return condition, cum_rewards, highst_levels, epi_probs, recover_rate

# %%
def run_condition(condition, num_episodes=100, num_actions=40):
    print(f"Running base condition: {condition}")

    # Initialize variables
    task = condition
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    # actions = np.full((num_episodes, num_actions), None)
    epi_probs = np.zeros(num_episodes)
    recover_rate = np.zeros(num_episodes)

    # Initialize the prior
    prior_alphas = np.full(len(norm_pairs), 0.001)
    prior_betas = np.full(len(norm_pairs), 0.001)

    # Prep for ground truth measure
    true_transitions =  np.array([task_func(task, pair) for pair in norm_pairs]).astype(int)

    # Run base PSRL agent
    for episode in range(num_episodes):

        sampled_probs = np.random.beta(prior_alphas, prior_betas)
        sampled_transitions = np.random.binomial(1, sampled_probs)
        epi_probs[episode] = sum(sampled_transitions)/len(sampled_transitions)
        recover_rate[episode] = (true_transitions == sampled_transitions).sum() / len(sampled_transitions)
        
    
        state_objs = norm_objs.copy()
        state_levels = np.zeros(len(norm_objs))

        reward = 0
        regeneratable = np.ones(len(norm_objs))

        max_level = 0
        for t in range(num_actions):
            action = policy(num_actions - t, state_objs, state_levels, sampled_transitions)
            if action is None:
                cum_rewards[episode, t] = cum_rewards[episode, t-1]
                highst_levels[episode, t] = highst_levels[episode, t-1]    
            else:
                #actions[episode, t] = action
                reward += get_reward(action)
                cum_rewards[episode, t] = max(cum_rewards[episode, t-1], reward)

                (_, state_objs, state_levels, regeneratable) = update_states(task, action, num_actions - t, state_objs, state_levels, regeneratable)

                if len(state_objs) > 0:
                    new_highest = np.max(state_levels)
                    max_level = max(max_level, new_highest)
                
                highst_levels[episode, t] = max_level

                if len(action) == 2:
                    [m, n] = action
                    norm_m = (m[0], m[1])
                    norm_n = (n[0], n[1])
                    pair_idx = norm_pairs.index((norm_m, norm_n))
                    

                    if task_func(task, (norm_m, norm_n)):
                        prior_alphas[pair_idx] += 1
                    else:
                        prior_betas[pair_idx] += 1
                    

    return condition, cum_rewards, highst_levels, epi_probs, recover_rate

# %%
# Plot the results
def plot_results(condition, prefix, cum_rewards, highst_levels, epi_probs, recover_rate):
    """ Plotting function runs only in the main process. """
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    axs[0].plot(np.max(cum_rewards, axis=0))
    axs[0].set_title(f"Cumulative Rewards ({condition})")
    
    axs[1].plot(np.max(highst_levels, axis=0))
    axs[1].set_title(f"Highest Levels ({condition})")
    
    axs[2].plot(epi_probs)
    axs[2].set_title(f"Episode Probabilities ({condition})")
    
    axs[3].plot(recover_rate)
    axs[3].set_title(f"Recovered True Transitions ({condition})")
    

    plt.tight_layout()
    plt.savefig(f"{prefix}_{condition}_results.png")
    print(f"✅ Saved plot for {prefix} {condition}")

# %%
condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_condition("simple", num_episodes=100, num_actions=40)
plot_results(condition, 'base', cum_rewards, highst_levels, epi_probs, recover_rate)

condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_condition("med", num_episodes=100, num_actions=40)
plot_results(condition, 'base', cum_rewards, highst_levels, epi_probs, recover_rate)


condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_condition("hard", num_episodes=100, num_actions=40)
plot_results(condition, 'base', cum_rewards, highst_levels, epi_probs, recover_rate)


# %%
condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_pcfg_condition("simple", num_episodes=50, num_actions=40)
plot_results(condition, 'pcfg', cum_rewards, highst_levels, epi_probs, recover_rate)

condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_pcfg_condition("med", num_episodes=50, num_actions=40)
plot_results(condition, 'pcfg', cum_rewards, highst_levels, epi_probs, recover_rate)

condition, cum_rewards, highst_levels, epi_probs, recover_rate = run_pcfg_condition("hard", num_episodes=10, num_actions=40)
plot_results(condition, 'pcfg', cum_rewards, highst_levels, epi_probs, recover_rate)



# %%
newrun = False
episodes = 50
max_retries = 5
if newrun:
    all_results = {}
    agents_to_run = ['base', 'pcfg']
else:
    agents_to_run = ['pcfg']

for agent in agents_to_run: 
    all_results[agent] = {}
    for condition in ["simple", "med", "hard"]:
        for attempt in range(max_retries):
            try:
                if agent == 'base':
                    results = run_condition(condition, episodes)
                elif agent == 'pcfg':
                    results = run_pcfg_condition(condition, episodes)
                
                all_results[agent][condition] = {
                    'condition': results[0],
                    'cum_rewards': results[1],
                    'highst_levels': results[2],
                    'epi_probs': results[3],
                    'recover_rate': results[4]
                }
                break
            
            except Exception as e:
                print(f"Error running {agent} on {condition} (attempt {attempt + 1}): {e}")
                traceback.print_exc()

                if attempt == max_retries - 1:
                    print(f"Max retries reached for {agent} on {condition}.")

measures = ['cum_rewards', 'highst_levels', 'epi_probs', 'recover_rate']
conditions = ['simple', 'med', 'hard']
agents = ['base', 'pcfg']
num_actions = 40

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharex=True)
for col, agent in enumerate(agents):
    for row, measure in enumerate(measures):
        ax = axes[row, col]
        for condition in conditions:
            data = all_results[agent][condition][measure]

            # Determine x-axis based on the measure
            if measure in ['cum_rewards', 'highst_levels']:
                # Use num_actions as the x-axis
                x_axis = np.arange(data.shape[1])  # Actions
                if measure == 'cum_rewards':
                    y_data = np.log1p(np.mean(data, axis=0))  # Log scale for rewards
                else:
                    y_data = np.mean(data, axis=0)  # Average across episodes
                
            
            else:
                # Use num_episodes as the x-axis
                x_axis = np.arange(data.shape[0])  # Episodes
                y_data = data  # Directly use the data
                ax.set_xlim(0, len(data))  # Set x-axis limit to the number of episodes
                if measure == 'epi_probs':
                    ax.set_ylim(0, 0.6)
                else:
                    ax.set_ylim(0.5, 1.1) 

            ax.plot(x_axis, y_data, label=condition)
            

        if col == 0:
            ax.set_ylabel(measure.replace('_', ' ').title(), fontsize=10)
        if row == 0:
            ax.set_title(f"{agent.title()} Agent", fontsize=12)
        if row < 2:
            ax.set_xlim(0, num_actions) 
            ax.set_xlabel("Action", fontsize=10)
        if row >= 2:
            ax.set_xlabel("Episode", fontsize=10)
        if row == 0 and col == 1:
            ax.legend(title="Condition", loc='upper right')

plt.tight_layout()
# plt.savefig(f"both_combined_results.png")




# %%
def run_all_conditions_parallel(agent='base'):
    conditions = ["simple", "med", "hard"]
    
    # Create a pool of workers and run the conditions in parallel
    with Pool(processes=3) as pool:
        if agent == 'base':
            results = pool.map(run_condition, conditions)
        elif agent == 'pcfg':
            results = pool.map(run_pcfg_condition, conditions)
        else:
            raise ValueError("Invalid agent type. Use 'base' or 'pcfg'.")

    # Unpack the results
    all_results = {}
    for i, condition in enumerate(conditions):
        all_results[condition] = {
            'condition': results[i][0],
            'cum_rewards': results[i][1],
            'highst_levels': results[i][2],
            'epi_probs': results[i][3],
            'recover_rate': results[i][4]
        }
    
    return all_results

# %%
def plot_combined_results(all_results, agent):
    """Plot all three conditions together in three separate graphs"""
    conditions = list(all_results.keys())
    
    # Define colors and line styles for each condition
    styles = {
        'simple': {'color': 'blue', 'linestyle': '-'},
        'med': {'color': 'green', 'linestyle': '--'},
        'hard': {'color': 'red', 'linestyle': ':'}
    }
    
    # Create three subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot rewards
    for condition in conditions:
        log_rewards = np.log1p(np.mean(all_results[condition]['cum_rewards'], axis=0))
        axs[0].plot(
            log_rewards,
            label=condition,
            color=styles[condition]['color'],
            linestyle=styles[condition]['linestyle']
        )
    axs[0].set_title("Cumulative Rewards (All Conditions)")
    axs[0].set_xlabel("Action")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    
    # Plot levels
    for condition in conditions:
        axs[1].plot(
            np.max(all_results[condition]['highst_levels'], axis=0),
            label=condition,
            color=styles[condition]['color'],
            linestyle=styles[condition]['linestyle']
        )
    axs[1].set_title("Highest Levels (All Conditions)")
    axs[1].set_xlabel("Action")
    axs[1].set_ylabel("Level")
    axs[1].legend()
    
    # Plot probabilities
    for condition in conditions:
        axs[2].plot(
            all_results[condition]['epi_probs'],
            label=condition,
            color=styles[condition]['color'],
            linestyle=styles[condition]['linestyle']
        )
    axs[2].set_title("Episode Probabilities (All Conditions)")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Probability")
    axs[2].legend()

    # Plot recovery rates
    for condition in conditions:
        axs[3].plot(
            all_results[condition]['recover_rate'],
            label=condition,
            color=styles[condition]['color'],
            linestyle=styles[condition]['linestyle']
        )
    axs[3].set_title("Recovered True Transitions (All Conditions)")
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("Recovery Rate")
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(f"{agent}_combined_results.png")
    print("✅ Saved combined plot")

# %%
# Execute the parallel runs and create the combined plot
if __name__ == "__main__":
    start_time = time.time()
    for agent in ['base', 'pcfg']:
        all_results = run_all_conditions_parallel(agent)
        plot_combined_results(all_results, agent)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
# %%
