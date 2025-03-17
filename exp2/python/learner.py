# %% 
# Load packages
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
def check_highest_level_improvable(objs_idx, state_objs, state_levels, sampled_transitions):
    if not objs_idx.size:
        return None

    obj = state_objs[objs_idx[0]]
    obj_level = state_levels[objs_idx[0]]

    matched_pairs = [pair for pair in norm_pairs if sampled_transitions[norm_pairs.index(pair)] == 1 and (pair[0] == obj or pair[1] == obj)]
    if not matched_pairs:
        # return check_highest_level_improvable(objs_idx[1:], state_objs, state_levels, sampled_transitions)
        return [ (obj[0], obj[1], obj_level) ]
    
    matched_objs = [pair[0] if pair[1] == obj else pair[1] for pair in matched_pairs]
    intersection = list(set(state_objs) & set(matched_objs))
    if not intersection:
        # return check_highest_level_improvable(objs_idx[1:], state_objs, state_levels, sampled_transitions)
        return [ (obj[0], obj[1], obj_level) ]
    
    chosen_obj = intersection[np.random.choice(range(len(intersection)))]
    state_objs_list = list(state_objs)
    chosen_obj_idx = state_objs_list.index(chosen_obj)
    return [(obj[0], obj[1], obj_level), (state_objs[chosen_obj_idx][0], state_objs[chosen_obj_idx][1], state_levels[chosen_obj_idx])]
    

def policy(actions_left, state_objs, state_levels, sampled_transitions):

    highest_level = np.max(state_levels)
    highest_level_obj_idx = np.where(state_levels == highest_level)[0]
    

    if actions_left < 2 or highest_level == len(colors):
        selected_obj = state_objs[np.random.choice(highest_level_obj_idx)]
        return [ (selected_obj[0], selected_obj[1], highest_level) ]
    
    else:
        # order objs by levels
        objs_idx_by_level = np.argsort(state_levels)[::-1]
        result = check_highest_level_improvable(objs_idx_by_level, state_objs, state_levels, sampled_transitions)
        if result is not None:
            return result
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
def run_condition(condition, num_episodes=1000, num_actions=40):
    print(f"Running condition: {condition}")

    # Initialize variables
    task = condition
    cum_rewards = np.zeros((num_episodes, num_actions))
    highst_levels = np.zeros((num_episodes, num_actions))
    actions = np.full((num_episodes, num_actions), None)
    epi_probs = np.zeros(num_episodes)

    # Initialize the prior
    prior_alphas = np.ones(len(norm_pairs))
    prior_betas = np.ones(len(norm_pairs))
    np.random.shuffle(norm_objs)

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
            reward += get_reward(action)
            cum_rewards[episode, t] = reward

            if action is None:
                break   
            else:
                #actions[episode, t] = action
                (_, state_objs, state_levels, regeneratable) = update_states(task, action, num_actions - t, state_objs, state_levels, regeneratable)

                new_highest_level = np.max(state_levels)
                if new_highest_level > max_level:
                    max_level = new_highest_level
                highst_levels[episode, t] = max_level

                if len(action) == 2:
                    [m, n] = action
                    norm_m = (m[0], m[1])
                    norm_n = (n[0], n[1])
                    pair_idx = norm_pairs.index((norm_m, norm_n))
                    
                    if task == 'simple' and simple_task((norm_m, norm_n)):
                        prior_alphas[pair_idx] += 1 * 10
                    
                    elif task == 'med' and med_task((norm_m, norm_n)):
                        prior_alphas[pair_idx] += 1 * 10

                    elif task == 'hard' and hard_task((norm_m, norm_n)):
                        prior_alphas[pair_idx] += 1 * 10
                    
                    else:
                        prior_betas[pair_idx] += 1 * 10

    return condition, cum_rewards, highst_levels, epi_probs


# %%
# Plot the results
def plot_results(condition, cum_rewards, highst_levels, epi_probs):
    """ Plotting function runs only in the main process. """
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(np.mean(cum_rewards, axis=0))
    axs[0].set_title(f"Cumulative Rewards ({condition})")
    
    axs[1].plot(np.mean(highst_levels, axis=0))
    axs[1].set_title(f"Highest Levels ({condition})")
    
    axs[2].plot(epi_probs)
    axs[2].set_title(f"Episode Probabilities ({condition})")
    
    plt.tight_layout()
    plt.savefig(f"{condition}_results.png")
    print(f"✅ Saved plot for {condition}")


# %%
conditions = ["simple", "med", "hard"]
plot_colors = ['blue', 'orange', 'green']
results = []

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_condition, cond): cond for cond in conditions}
        for future in tqdm(as_completed(futures), total=len(conditions), desc="Processing Conditions"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"❌ Error in condition {futures[future]}: {e}")

        # After parallel work, plot in main thread
        # for condition, cum_rewards, highst_levels, epi_probs in results:
        #     plot_results(condition, cum_rewards, highst_levels, epi_probs)

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot cumulative rewards in log scale
ax1 = axes[0]
for i, (condition, cum_rewards, _, _) in enumerate(results):
    mean_rewards = np.mean(cum_rewards+1, axis=0)  # Mean across episodes
    print(mean_rewards)
    ax1.plot(mean_rewards, label=condition, color=plot_colors[i])
ax1.set_yscale('log')
ax1.set_title("Cumulative Rewards (Log Scale)")
ax1.set_xlabel("Action")
ax1.set_ylabel("Cumulative Rewards (Log)")
ax1.legend()

# Plot highest levels
ax2 = axes[1]
for i, (condition, _, highst_levels, _) in enumerate(results):
    mean_highst_levels = np.mean(highst_levels, axis=0)  # Mean across episodes
    ax2.plot(mean_highst_levels, label=condition, color=plot_colors[i])
ax2.set_title("Highest Levels")
ax2.set_xlabel("Action")
ax2.set_ylabel("Highest Levels")
ax2.legend()

# Plot episode probabilities
ax3 = axes[2]
for i, (condition, _, _, epi_probs) in enumerate(results):
    ax3.plot(epi_probs, label=condition, color=plot_colors[i])
ax3.set_title("Episode Probabilities")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Probability")
ax3.legend()

# Layout adjustment to avoid overlap
plt.tight_layout()

# Save the figure automatically
plt.savefig("conditions_plots.png")
plt.show()

# %%
