# %%
import numpy as np
from itertools import combinations

# %%
colors = [0, 1, 2] # up to 5 (six values) in actual experiment
max_color = max(colors)

shapes = [0, 1] # up to 3 (four values) in actual experiment
textures = [0, 1] # up to 3 (four values) in n actual experiment

num_actions = 10 # num_actions = 40 in actual experiment

base_items = [ f"{s}{t}0" for s in shapes for t in textures]
all_items = [ f"{s}{t}{c}" for s in shapes for t in textures for c in colors]

# Get all states
all_states = []
all_states.append(frozenset())
for i in range(1, len(base_items)+1):
    for combo in combinations(all_items, i):
        state = frozenset(combo)
        all_states.append(state)


# Get all actions
combination_actions = list(combinations(all_items, 2))
consume_actions = [(item,) for item in all_items]
all_actions = combination_actions + consume_actions

# Prep mappings
state_to_idx = {state: idx for idx, state in enumerate(all_states)}
idx_to_state = {idx: state for idx, state in enumerate(all_states)}
action_to_idx = {action: idx for idx, action in enumerate(all_actions)}
idx_to_action = {idx: action for idx, action in enumerate(all_actions)}

# Initial state
initial_states = [frozenset(base_items)]
initial_state_idx = state_to_idx[initial_states[0]]

# Print number of states and actions
num_states = len(all_states)
num_actions = len(all_actions)
print(f"Number of states: {num_states}") #794
print(f"Number of actions: {num_actions}") #78

# %%
# Item to proporties
def parse_item(item):
    if not item or len(item) < 3:
        return None, None, None
    return int(item[0]), int(item[1]), int(item[2:])

# Combination rule
def combine_items(item1, item2, rule='simple'):
    shape1, texture1, color1 = parse_item(item1)
    shape2, texture2, color2 = parse_item(item2)
    
    new_color = max(color1, color2) + 1    
    if new_color > max_color:
        return None

    if rule == 'simple' and shape1 == shape2:
        new_item = f"{shape1}{texture2}{new_color}"
        return new_item

    if rule == 'med' and shape1 + shape2 == 3 and texture1 != texture2:
        new_item = f"{shape1}{texture2}{new_color}"
        return new_item

    if rule == 'hard' and shape1 + shape2 == 3 and texture1 >= texture2:
        new_item = f"{shape1}{texture2}{new_color}"
        return new_item

    return None

# Reward function
def get_reward(state, action):
    if len(action) == 1:
        # check if item is present
        item = action[0]
        if item in state:
            shape, texture, color = parse_item(item)
            return 10 ** color
        return 0
    return 0
        
# Transition probabilities
def get_transition_probs(state, action, rule='simple'):
    
    if len(action) == 1:
        item = action[0]
        if item not in state:
            return {frozenset(state): 1.0}
        else:
            next_state = set(state) - {item}
            return {frozenset(next_state): 1.0}
    
    else:
        item1, item2 = action
        if item1 not in state or item2 not in state:
            return {frozenset(state): 1.0} 
    
        # Try to combine items
        combined_item = combine_items(item1, item2, rule)
        if combined_item is None:
            return {frozenset(state): 1.0}
        else:
            # Create the new state after successful combination
            success_state = set(state) - {item1, item2}
            success_state.add(combined_item)
            return { frozenset(success_state): 1.0 }
        
# %%
reward_matrix = np.zeros((num_states, num_actions))
transition_matrix = np.zeros((num_states * num_actions, num_states))

for s_idx, state in idx_to_state.items():
    for a_idx, action in idx_to_action.items():
        sa_idx = s_idx * num_actions + a_idx
            
        # Get reward for this state-action pair
        reward_matrix[s_idx, a_idx] = get_reward(state, action)
            
        # Get transition probabilities
        transition_probs = get_transition_probs(state, action)
            
        # Fill in transition matrix
        for next_state, prob in transition_probs.items():
            if next_state in state_to_idx:
                next_s_idx = state_to_idx[next_state]
                transition_matrix[sa_idx, next_s_idx] = prob


# %%

