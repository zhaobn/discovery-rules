# %% 
# Load packages
import numpy as np

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
        return check_highest_level_improvable(objs_idx[1:], state_objs, state_levels, sampled_transitions)
    
    matched_objs = [pair[0] if pair[1] == obj else pair[1] for pair in matched_pairs]
    intersection = list(set(state_objs) & set(matched_objs))
    if not intersection:
        return check_highest_level_improvable(objs_idx[1:], state_objs, state_levels, sampled_transitions)

    else:
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


def update_states(action, action_left, state_objs, state_levels, regeneratable):

    action_left = action_left - 1
    if action is None or action_left < 0:
        return None

    state_objs_list = list(state_objs)
    if len(action) == 2:
        [m, n] = action
        norm_m = (m[0], m[1])
        norm_n = (n[0], n[1])
        is_valid = simple_task((norm_m, norm_n))
        
        if is_valid:
            
            new_obj = (norm_m[0], norm_n[1])
            new_level = m[2] + n[2] + 1
            
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
num_episodes = 30
num_actions = 20

cum_rewards = np.zeros((num_episodes, num_actions))
highst_levels = np.zeros((num_episodes, num_actions))
actions = np.full((num_episodes, num_actions), None)

# initialize the prior
prior_alphas = np.ones(len(norm_pairs))
prior_betas = np.ones(len(norm_pairs))
np.random.shuffle(norm_objs)

for episode in range(num_episodes):

    sampled_probs = np.random.beta(prior_alphas, prior_betas)
    sampled_transitions = np.random.binomial(1, sampled_probs)
    
    state_objs = norm_objs.copy()
    state_levels = np.zeros(len(norm_objs))

    reward = 0
    regeneratable = np.ones(len(norm_objs))

    for t in range(num_actions):
        action = policy(num_actions - t, state_objs, state_levels, sampled_transitions)
        reward += get_reward(action)
        cum_rewards[episode, t] = reward

        if action is None:
            break   
        else:
            actions[episode, t] = action
            (_, state_objs, state_levels, regeneratable) = update_states(action, num_actions - t, state_objs, state_levels, regeneratable)
            
            highst_levels[episode, t] = np.max(state_levels)

            if len(action) == 2:
                [m, n] = action
                norm_m = (m[0], m[1])
                norm_n = (n[0], n[1])
                pair_idx = norm_pairs.index((norm_m, norm_n))
                if simple_task((norm_m, norm_n)):
                    prior_alphas[pair_idx] += 1
                else:
                    prior_betas[pair_idx] += 1


# %%
