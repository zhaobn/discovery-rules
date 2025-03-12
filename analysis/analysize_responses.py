# %%
import os
import openai
import pandas as pd
import json

openai.api_key = os.getenv("SECRET_KEY_DISCOVERY_GAME")

# %%
def process_row(response_1, response_2, model="gpt-4"):
  prompt = f"""
You will be provided with two participant responses. Your job is to summarize the information based on the following criteria:

1. Tips: Sentences that explain how to interact with the game or navigate its mechanics. This includes:
    * Instructions about picking up, dropping, or moving items.
    * Insights about how color changes impact gameplay.
    * Generic observations that higher points are better.
    * Suggestions to use game features like hints or visual aids.

2. Patterns: Sentences that provide insights into decision strategies or relationships between item features (e.g., shape, texture, same, different) that affect combinations or outcomes.
    * Focus specifically on patterns or rules .

3. NA: Sentences that indicate confusion, lack of understanding, or the inability to identify meaningful rules or patterns.
    * Examples: Expressions of randomness, frustration, or lack of clarity about the game.

Sort each sentence in the responses into these categories. Respond in a JSON object. Use an empty list if no sentence satisfies a criterion. Use a list of strings for each sentence that satisfies a criterion.

Additionally, add a patterns_summary entry to the json object, and summarize the patterns as *accurate* and *succinct* as you can.

Response 1: {response_1}
Response 2: {response_2}

Sort these sentences according to the criteria and respond in the requested JSON format.
"""

  try:
    response = openai.chat.completions.create(
      model=model,
      messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
  except Exception as e:
    print(f"Error processing row: {e}")
    return None

# %% Test on a single row
response_data = pd.read_csv("../data/message_data.csv")
response_data

row = response_data[response_data["id"] == 8]
response_1 = row["messageHow"].values[0]
response_2 = row["messageRules"].values[0]


processed_result = process_row(response_1, response_2)
print(processed_result)

# %%
# Process all respones
structured_outputs = []
for index, row in response_data.iterrows():
  response_1 = row["messageHow"]
  response_2 = row["messageRules"]

  sorted = process_row(response_1, response_2)
  sorted_string = json.dumps(sorted)

  structured_outputs.append(sorted_string)

  # Print progress
  print(f"Processed row {index + 1}/{len(response_data)}")

# Save the DataFrame to a new CSV
response_data["coded"] = structured_outputs
response_data.to_csv("../data/responses_processed.csv", index=False)


# %% Analyze the processed data
response_data = pd.read_csv("../data/responses_processed.csv")

def decode_coded_string(row):
    return json.loads(row['coded'])

response_data['decoded'] = response_data.apply(decode_coded_string, axis=1)
response_data['decoded'] = response_data['decoded'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Create new columns based on the extracted JSON data
response_data['n_tips'] = response_data['decoded'].apply(lambda x: len(x.get('Tips', [])) if 'Tips' in x else len(x.get('tips', [])))
response_data['len_tips'] = response_data['decoded'].apply(lambda x: sum(len(tip) for tip in x.get('Tips', [])) if 'Tips' in x else sum(len(tip) for tip in x.get('tips', [])))

response_data['n_rules'] = response_data['decoded'].apply(lambda x: len(x.get('Patterns', [])) if 'Patterns' in x else len(x.get('patterns', [])))
response_data['len_rules'] = response_data['decoded'].apply(lambda x: sum(len(rule) for rule in x.get('Patterns', [])) if 'Patterns' in x else sum(len(rule) for rule in x.get('patterns', [])))

response_data['n_NAs'] = response_data['decoded'].apply(lambda x: len(x.get('NA', [])) if 'NA' in x else len(x.get('na', [])))
response_data['len_NAs'] = response_data['decoded'].apply(lambda x: sum(len(na) for na in x.get('NA', [])) if 'NA' in x else sum(len(na) for na in x.get('na', [])))

response_data['len_summary'] = response_data['decoded'].apply(lambda x: len(x['patterns_summary']))

# Drop the decoded column if no longer needed
response_data.drop(columns=['decoded'], inplace=True)
response_data.to_csv("../data/responses_processed.csv", index=False)



# %%




# %%
# https://github.com/openai/openai-python
# completion = openai.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": "How do I output all files in a directory using Python?",
#         },
#     ],
# )
# print(completion.choices[0].message.content)
