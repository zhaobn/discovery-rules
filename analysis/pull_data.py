# %%
import pandas as pd
import numpy as np
import json

# %%
file_path = "../data/grid_pilot_1.tsv"

data = pd.read_csv(file_path, sep="\t")
data = data[data['worker'].str.len() > 10]
#data = data[data['assignment'] == 'hard-1']

data['subject'] = data['subject'].str.replace(r'{\\prolific', '"{\\"prolific', regex=True)
data['actions'] = data['actions'].str.replace(r'\\\\', r'\\', regex=True)
data['actions'] = data['actions'].str.replace(r'{\\act-1', '"{\\\"act-1', regex=True)
data['events'] = data['events'].str.replace(r'{\\event-1', '"{\\\"event-1', regex=True)


# %%
batch_name = 'pilot_newhard_full'

data['subject_parsed'] = data['subject'].apply(json.loads)

def extract_json_fields(json_str):
  try:
    data = json.loads(json_str)
    return pd.Series(data)
  except json.JSONDecodeError:
    return pd.Series([None] * len(data.columns))

subject_data = data['subject_parsed'].apply(extract_json_fields)
subject_full_data = pd.concat([data[['id', 'assignment']], subject_data], axis=1)

prolific_ids = subject_full_data[['id', 'prolific_id']]
prolific_ids.to_csv(f"../data/{batch_name}_id_data.csv", index=False)

#prolific_ids.shape[0]

# %% calc bonus
bonus_data = subject_data[['prolific_id', 'total_points']]
bonus_data = bonus_data[bonus_data['total_points'] > 0]

def calculate_bonus(points, max_digits, min_bonus=0, max_bonus=1):
  num_digits = len(str(int(points))) if points > 0 else 0
  normalized_points = num_digits / max_digits
  bonus = min_bonus + (max_bonus - min_bonus) * normalized_points
  return round(bonus, 2)

max_digits = len(str(int(bonus_data['total_points'].max())))
bonus_data['bonus'] = bonus_data['total_points'].apply(lambda x: calculate_bonus(x, max_digits))

bonus_data[['prolific_id', 'bonus']].to_csv(f"../data/{batch_name}_bonus_data.csv", index=False)
bonus_data['bonus'].mean()

# %% output self reports
message_data = subject_full_data[['id', 'condition', 'messageHow', 'messageRules', 'total_points']]
message_data.to_csv(f"../data/{batch_name}_message_data.csv", index=False)


# %% save anonymized data
subject_full_data.reset_index(drop=True, inplace=True)
subject_full_data.drop(columns=['prolific_id', 'allow_regeneration'], inplace=True)
subject_full_data.to_csv(f"../data/{batch_name}_subject_data.csv", index=False)


# %%

data['actions_parsed'] = data['actions'].apply(json.loads)

def extract_actions(actions_str):
  actions_dict = json.loads(actions_str)
  actions_list = []
  for action_id, action_data in actions_dict.items():
    action_data['action_id'] = int(action_id.replace('act-', ''))
    actions_list.append(action_data)
  return actions_list

expanded_actions = data['actions_parsed'].apply(extract_actions)
expanded_actions_df = pd.json_normalize(expanded_actions.explode())

id_token = subject_full_data[['id', 'token', 'assignment']]
merged_df = expanded_actions_df.merge(id_token, on='token', how='left')

cols = ['id', 'action_id'] + [col for col in merged_df.columns if col not in ['id', 'action_id']]
merged_df = merged_df[cols]
merged_df.to_csv(f"../data/{batch_name}_action_data.csv", index=False)


# %%

data['events_parsed'] = data['events'].apply(json.loads)

def extract_event_details(event_str):
  event_data = json.loads(event_str)
  details = []
  for event_id, event_info in event_data.items():
    details.append({
      'timestamp': event_info['timestamp'],
      'event_id': int(event_id.replace('event-', '')),
      'action': event_info['action'],
      'x': event_info['x'],
      'y': event_info['y'],
      'actionsLeft': event_info['actionsLeft'],
      'currentPoints': event_info['currentPoints'],
      'currentlyCarrying': event_info['currentlyCarrying'],
      'token': event_info['token']
    })
  return details

expanded_data = data['events_parsed'].apply(extract_event_details)
expanded_df = pd.json_normalize(expanded_data.explode())

merged_df = expanded_df.merge(id_token, on='token', how='left')
cols = ['id', 'event_id'] + [col for col in merged_df.columns if col not in ['id', 'event_id']]
merged_df = merged_df[cols]
merged_df.to_csv(f"../data/{batch_name}_events_data.csv", index=False)



# %% Pilot Jan 13 has different message columns
p3_data = data[data['version']==0.4]
subject_data = p3_data['subject_parsed'].apply(extract_json_fields)
subject_full_data = pd.concat([p3_data[['id', 'assignment']], subject_data], axis=1)

subject_full_data.reset_index(drop=True, inplace=True)
subject_full_data.to_csv(f"../data/{batch_name}_subject_data.csv", index=False)

# %%
expanded_actions = p3_data['actions_parsed'].apply(extract_actions)
expanded_actions_df = pd.json_normalize(expanded_actions.explode())

id_token = subject_full_data[['id', 'token', 'assignment', 'version']]
merged_df = expanded_actions_df.merge(id_token, on='token', how='left')

cols = ['id', 'action_id'] + [col for col in merged_df.columns if col not in ['id', 'action_id']]
merged_df = merged_df[cols]
merged_df.to_csv("../data/action_data.csv", index=False)
# %%
subject_data = pd.read_csv("../data/subject_data.csv")
bonus_data = subject_data[['prolific_id', 'total_points', 'version', 'assignment']]
bonus_data = bonus_data[bonus_data['total_points'] > 0]

max_digits = len(str(int(bonus_data['total_points'].max())))
bonus_data['bonus'] = bonus_data['total_points'].apply(lambda x: calculate_bonus(x, max_digits))

bonus_data[['prolific_id', 'bonus']].to_csv("../data/bonus_data.csv", index=False)
# %%
