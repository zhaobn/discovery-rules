# %% 
# Load packages
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


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


# %%
# Base PSRL agent
def find_combination(state_objs, state_levels, sampled_transitions):
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
                            and (pair[0] == obj or pair[1] == obj)]            
            if matched_pairs:
                matched_objs = [pair[0] if pair[1] == obj else pair[1] for pair in matched_pairs]
                intersection = [matched_obj for matched_obj in matched_objs if matched_obj in state_objs]
                
                if intersection:
                    chosen_obj = intersection[np.random.choice(len(intersection))]
                    chosen_obj_idx = state_objs.index(chosen_obj)

                    return [(state_objs[obj_idx][0], state_objs[obj_idx][1], state_levels[obj_idx]), 
                            (state_objs[chosen_obj_idx][0], state_objs[chosen_obj_idx][1], state_levels[chosen_obj_idx])]
        
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

    state_objs_list = list(state_objs)
    if len(action) == 2:
        [m, n] = action
        norm_m = (m[0], m[1])
        norm_n = (n[0], n[1])
        
        is_valid = False
        if task == 'simple':
            is_valid = simple_task((norm_m, norm_n))
        if task == 'med':
            is_valid = med_task((norm_m, norm_n))
        if task == 'hard':
            is_valid = hard_task((norm_m, norm_n))

        
        if is_valid:
            new_obj = (norm_m[0], norm_n[1])
            
            # Find indices where the object matches m and n
            m_indices = [i for i, (obj, level) in enumerate(zip(state_objs_list, state_levels)) 
                 if obj == norm_m and level == m[2]]
            n_indices = [i for i, (obj, level) in enumerate(zip(state_objs_list, state_levels)) 
                 if obj == norm_n and level == n[2]]
            
            if m_indices and n_indices:
                m_idx = m_indices[0]
                n_idx = n_indices[0]

                if regeneratable[m_idx] == 1 and regeneratable[n_idx] == 1:
                    regeneratable[m_idx] = 0
                    regeneratable[n_idx] = 0
                
                if regeneratable[m_idx] == 1 and  regeneratable[n_idx] == 0:
                    regeneratable[m_idx] = 0
                    new_state_objs = np.delete(state_objs, [n_idx], axis=0)
                    new_state_levels = np.delete(state_levels, [n_idx], axis=0)

                if regeneratable[m_idx] == 0 and regeneratable[n_idx] == 1:
                    regeneratable[n_idx] = 0
                    new_state_objs = np.delete(state_objs, [m_idx], axis=0)
                    new_state_levels = np.delete(state_levels, [m_idx], axis=0)

                if regeneratable[m_idx] == 0 and regeneratable[n_idx] == 0:
                    new_state_objs = np.delete(state_objs, [m_idx, n_idx], axis=0)
                    new_state_levels = np.delete(state_levels, [m_idx, n_idx], axis=0)

                # Add new_obj and new_level to state_objs and state_levels
                new_state_objs = np.append(new_state_objs, [new_obj], axis=0)
                new_level = np.max([m[2], n[2]]) + 1
                new_state_levels = np.append(new_state_levels, new_level)
                regeneratable = np.append(regeneratable, 0)
        
        else:   
            return (action_left, state_objs, state_levels, regeneratable)
    
    else:
        (m, n, l) = action[0]
        obj_indices = [i for i, (obj, level) in enumerate(zip(state_objs_list, state_levels)) 
                 if obj == (m, n) and level == l]
        obj_idx = obj_indices[0]

        if regeneratable[obj_idx] == 1:
            regeneratable[obj_idx] = 0
            new_state_objs = state_objs
            new_state_levels = state_levels
        else:
            new_state_objs = np.delete(state_objs, [obj_idx], axis=0)
            new_state_levels = np.delete(state_levels, [obj_idx], axis=0)

    new_state_objs = [(i,j) for i,j in new_state_objs]

    return (action_left, new_state_objs, new_state_levels, regeneratable)

# %%
def run_condition(condition, num_episodes=10, num_actions=40):
    print(f"Running condition: {condition}")

    # Initialize variables
    task = condition
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    # actions = np.full((num_episodes, num_actions), None)
    epi_probs = np.zeros(num_episodes)

    # Initialize the prior
    prior_alphas = np.full(len(norm_pairs), 0.001)
    prior_betas = np.full(len(norm_pairs), 0.001)

    # Run base PSRL agent
    for episode in range(num_episodes):

        sampled_probs = np.random.beta(prior_alphas, prior_betas)
        sampled_transitions = np.random.binomial(1, sampled_probs)
        epi_probs[episode] = sum(sampled_transitions)/len(sampled_transitions)
    
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
                    
                    if task == 'simple':
                        if simple_task((norm_m, norm_n)):
                            prior_alphas[pair_idx] += 1
                        else:
                            prior_betas[pair_idx] += 1
                    
                    if task == 'med':
                        if med_task((norm_m, norm_n)):
                            prior_alphas[pair_idx] += 1
                        else:
                            prior_betas[pair_idx] += 1

                    if task == 'hard':
                        if hard_task((norm_m, norm_n)):
                            prior_alphas[pair_idx] += 1
                        else:
                            prior_betas[pair_idx] += 1

    return condition, cum_rewards, highst_levels, epi_probs

# %%
# Plot the results
def plot_results(condition, cum_rewards, highst_levels, epi_probs):
    """ Plotting function runs only in the main process. """
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(np.max(cum_rewards, axis=0))
    axs[0].set_title(f"Cumulative Rewards ({condition})")
    
    axs[1].plot(np.max(highst_levels, axis=0))
    axs[1].set_title(f"Highest Levels ({condition})")
    
    axs[2].plot(epi_probs)
    axs[2].set_title(f"Episode Probabilities ({condition})")
    
    plt.tight_layout()
    plt.savefig(f"{condition}_results.png")
    print(f"✅ Saved plot for {condition}")



# %%
condition, cum_rewards, highst_levels, epi_probs = run_condition("simple", num_episodes=100, num_actions=40)
plot_results("simple", cum_rewards, highst_levels, epi_probs)

condition, cum_rewards, highst_levels, epi_probs = run_condition("med", num_episodes=100, num_actions=40)
plot_results(condition, cum_rewards, highst_levels, epi_probs)


condition, cum_rewards, highst_levels, epi_probs = run_condition("hard", num_episodes=100, num_actions=40)
plot_results(condition, cum_rewards, highst_levels, epi_probs)


# %%
def run_single_condition(condition):
    result = run_condition(condition, num_episodes=400, num_actions=40)
    return result

def run_all_conditions_parallel():
    conditions = ["simple", "med", "hard"]
    
    # Create a pool of workers and run the conditions in parallel
    with Pool(processes=3) as pool:
        results = pool.map(run_single_condition, conditions)
    
    # Unpack the results
    all_results = {}
    for i, condition in enumerate(conditions):
        all_results[condition] = {
            'condition': results[i][0],
            'cum_rewards': results[i][1],
            'highst_levels': results[i][2],
            'epi_probs': results[i][3]
        }
        
        # Save individual plots
        plot_results(
            condition,
            all_results[condition]['cum_rewards'],
            all_results[condition]['highst_levels'],
            all_results[condition]['epi_probs']
        )
    
    return all_results

# %%
def plot_combined_results(all_results):
    """Plot all three conditions together in three separate graphs"""
    conditions = list(all_results.keys())
    
    # Define colors and line styles for each condition
    styles = {
        'simple': {'color': 'blue', 'linestyle': '-'},
        'med': {'color': 'green', 'linestyle': '--'},
        'hard': {'color': 'red', 'linestyle': ':'}
    }
    
    # Create three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
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
    axs[0].set_xlabel("Episode")
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
    axs[1].set_xlabel("Episode")
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
    
    plt.tight_layout()
    plt.savefig("combined_results.png")
    print("✅ Saved combined plot")

# %%
# Execute the parallel runs and create the combined plot
if __name__ == "__main__":
    start_time = time.time()
    all_results = run_all_conditions_parallel()
    plot_combined_results(all_results)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
# %%
