# %%
from random import sample
from math import log
import numpy as np
import pandas as pd

# %%
class Rational_rules:
  def __init__(self, p_rules, cap=10):
    self.NON_TERMINALS = [x[0] for x in p_rules]
    self.PRODUCTIONS = {}
    self.CAP = cap
    for rule in p_rules:
      self.PRODUCTIONS[rule[0]] = rule[1]

  def generate_tree(self, logging=True, tree_str='S', log_prob=0., depth=0):
    current_nt_indices = [tree_str.find(nt) for nt in self.NON_TERMINALS]
    # Sample a non-terminal for generation
    to_gen_idx = sample([idx for idx, el in enumerate(current_nt_indices) if el > -1], 1)[0]
    to_gen_nt = self.NON_TERMINALS[to_gen_idx]
    # Do generation
    leaf = sample(self.PRODUCTIONS[to_gen_nt], 1)[0]
    to_gen_tree_idx = tree_str.find(to_gen_nt)
    tree_str = tree_str[:to_gen_tree_idx] + leaf + tree_str[(to_gen_tree_idx+1):]
    # Update production log prob
    log_prob += log(1/len(self.PRODUCTIONS[to_gen_nt]))
    # Increase depth count
    depth += 1

    # Recursively rewrite string
    if any (nt in tree_str for nt in self.NON_TERMINALS) and depth <= self.CAP:
      return self.generate_tree(logging, tree_str, log_prob, depth)
    elif any (nt in tree_str for nt in self.NON_TERMINALS):
      if logging:
        print('====DEPTH EXCEEDED!====')
      return None
    else:
      if logging:
        print(tree_str, log_prob)
      return tree_str, log_prob

  @staticmethod
  def evaluate(rule, data):
    d = data
    pred = eval(rule[0])
    likelihood = (int(pred)==int(d[2]))
    return likelihood, rule[1]

  @staticmethod
  def predict(rule, data):
    d = data
    pred = eval(rule[0])
    pred = 0 if pred < 0 else pred
    pred = 16 if pred > 16 else pred
    return str(pred), rule[1]


# %%
# Evaluation functions
def fand(x, y): return x and y
def fdisj(x, y): return x or y

def ftriangle(): return 0
def fcircle(): return 1
def fsquare(): return 2
def fdiamond(): return 3

def fplain(): return 0
def fcheckered(): return 1
def fstripes(): return 2
def fdots(): return 3

# Example data ((0,0),(1,1))
def sameshape(d): return d[0][0] == d[1][0]
def diffshape(d): return d[0][0] != d[1][0]

def sametexture(d): return d[0][1] == d[1][1]
def difftexture(d): return d[0][1] != d[1][1]

def pairshape(d, val1, val2): return (d[0][0] == val1 and d[1][0] == val2) or (d[0][0] == val2 and d[1][0] == val1)
def pairtexture(d, val1, val2): return (d[0][1] == val1 and d[1][1] == val2) or (d[0][1] == val2 and d[1][1] == val1)

def obj_1(d): return d[0]
def obj_2(d): return d[1]

def isshape(obj, val): return obj[0] == val
def istexture(obj, val): return obj[1] == val

# %% 
productions = [
  ['S', ['A', 'B', 'fand(A,B)']],
  ['A', ['sameshape(d)', 'diffshape(d)', 'C', 'fdisj(C,C)']],
  ['B', ['sametexture(d)', 'difftexture(d)', 'D', 'fdisj(D,D)']],
  ['C', ['isshape(obj_1(d),X)', 'isshape(obj_2(d),X)', 'fand(isshape(obj_1(d),X),isshape(obj_2(d),X))', 'pairshape(d,X,X)']],
  ['D', ['istexture(obj_1(d),Y)', 'istexture(obj_2(d),Y)', 'fand(istexture(obj_1(d),Y),istexture(obj_2(d),Y))', 'pairtexture(d,Y,Y)']],
  ['X', ["ftriangle()", "fcircle()", "fsquare()", "fdiamond()"]],
  ['Y', ["fplain()", "fcheckered()", "fstripes()", "fdots()"]],
]
test = Rational_rules(productions, cap=100)

# %% Debug
# x = test.generate_tree()
# x
# d = ((0, 0), (0, 0))
# eval(x[0])

# %% 
# First, generate a lot of trees and save them to a table
generator = Rational_rules(productions, cap=1000)
results = {}

for _ in range(100000):
  (generated_string, log_prob) = generator.generate_tree(logging=False)
  
  if generated_string is not None:
    
    if generated_string in results:
      # String exists, update its values
      current_avg_prob, current_count = results[generated_string]
      new_avg_prob = ((current_avg_prob * current_count) + log_prob) / (current_count + 1)
      results[generated_string] = [
          new_avg_prob,
          current_count + 1
      ]
    
    else:
      # New string, initialize its values
      results[generated_string] = [log_prob, 1]

# Convert results to pandas DataFrame
data = [
    {"string": string, "log_prob": avg_log_prob, "count": count}
    for string, (avg_log_prob, count) in results.items()
]
df = pd.DataFrame(data)
df = df.sort_values(by=["count", "log_prob"], ascending=[False, False]).reset_index(drop=True)

df.to_csv('tree_prob.csv', index=False)

# %% 
# Next, get MDPs for each tree
shapes = np.arange(4)   # ["triangle", "circle", "square", "diamond"]
textures = np.arange(4) # ["plain", "checkered", "stripes", "dots"]
norm_objs = [(s, t) for s in shapes for t in textures]
norm_pairs = [(m, n) for m in norm_objs for n in norm_objs if m != n]


# %%
df = pd.read_csv('tree_prob.csv')
test_df = df.head(10).copy()  # Copy the first 10 rows for testing
test_pairs = norm_pairs[:5]  # Use the first 5 pairs for testing

def functions_to_transitions(test_df, test_pairs):
  num_pairs = len(test_pairs)
  num_rows = len(test_df)
    
  # Use array to store results
  result_array = np.zeros((num_rows, num_pairs), dtype=int)
    
  for i in range(num_pairs):
    d = test_pairs[i]
      
    # For each row, evaluate the string using each pair
    for idx, (j, row) in enumerate(test_df.iterrows()):
      expr = row['string']
      
      try:
        result = eval(expr)
        result_array[idx, i] = int(bool(result))
      except Exception as e:
        print(f"Error evaluating expression: {expr}, Error: {e}")
        result_array[idx, i] = -1
  
    # Create a DataFrame with the results
    result_df = pd.DataFrame(result_array, columns=[f'pair_{i}' for i in range(num_pairs)])
    final_df = pd.concat([test_df.reset_index(drop=True), result_df], axis=1)
  
  return final_df


new_df = functions_to_transitions(df, norm_pairs)
new_df.to_csv('tree_mdps.csv', index=False)


# %%
# out of curiosity, check if the ground truths are covered
def simple_task (pair):
    (m, n) = pair
    return m[0] == n[0]

def med_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] != n[1]

def hard_task (pair):
    (m, n) = pair
    return m[0] + n[0] == 3 and m[1] >= n[1]

simple_mdp = [simple_task(pair) for pair in norm_pairs]
med_mdp = [med_task(pair) for pair in norm_pairs]
hard_mdp = [hard_task(pair) for pair in norm_pairs]

def check_ground_truth(df, ground_truth):
  for _, row in df.iterrows():
    if all(row[f'pair_{j}'] == ground_truth[j] for j in range(len(ground_truth))):
      return True
  
  return False

simple_covered = check_ground_truth(new_df, simple_mdp)
med_covered = check_ground_truth(new_df, med_mdp)
hard_covered = check_ground_truth(new_df, hard_mdp)

# %%
