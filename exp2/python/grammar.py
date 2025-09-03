# %%
from random import sample
from math import log
import numpy as np
import pandas as pd

# %%
class Mental_grammar:
  def __init__(self, p_rules, cap=10):
    self.NON_TERMINALS = [x[0] for x in p_rules]
    self.PRODUCTIONS = {}
    self.PROBABILITIES = {}
    self.CAP = cap

    for rule in p_rules:
      nt = rule[0]
      # Check if rules have probabilities
      if isinstance(rule[1][0], tuple):
        # Format: [('rule', prob), ...]
        self.PRODUCTIONS[nt] = [item[0] for item in rule[1]]
        self.PROBABILITIES[nt] = [item[1] for item in rule[1]]
      else:
        # Old format: ['rule1', 'rule2', ...]
        self.PRODUCTIONS[nt] = rule[1]
        self.PROBABILITIES[nt] = [1/len(rule[1])] * len(rule[1])  # uniform
      
  def generate_tree(self, logging=True, tree_str='S', log_prob=0., depth=0):
    current_nt_indices = [tree_str.find(nt) for nt in self.NON_TERMINALS]
    # Sample a non-terminal for generation
    to_gen_idx = sample([idx for idx, el in enumerate(current_nt_indices) if el > -1], 1)[0]
    to_gen_nt = self.NON_TERMINALS[to_gen_idx]
    
    # Do generation
    productions = self.PRODUCTIONS[to_gen_nt]
    probs = self.PROBABILITIES[to_gen_nt]
    leaf_idx = np.random.choice(len(productions), p=probs)
    leaf = productions[leaf_idx]

    to_gen_tree_idx = tree_str.find(to_gen_nt)
    tree_str = tree_str[:to_gen_tree_idx] + leaf + tree_str[(to_gen_tree_idx+1):]
  
    # Update production log prob
    log_prob += np.log(probs[leaf_idx])
    
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


# %%
# Evaluation functions
def fand(x, y): return x and y
def fnot(x): return not x
def f_or(x, y): return x or y

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
def sametexture(d): return d[0][1] == d[1][1]

def isshape(obj, val): return obj[0] == val
def istexture(obj, val): return obj[1] == val

def pairshape(d, val1, val2): return (d[0][0] == val1 and d[1][0] == val2) or (d[1][0] == val1 and d[0][0] == val2)
def pairtexture(d, val1, val2): return (d[0][1] == val1 and d[1][1] == val2) or (d[1][1] == val1 and d[0][1] == val2)

def obj_1(d): return d[0]
def obj_2(d): return d[1]


# %% 
# productions = [
#   ['S', ['A', 'B', 'fand(S,S)', 'fnot(S)']],
#   ['A', ['C', 'sameshape(d)', 'isshape(D,X)']],
#   ['B', ['C', 'sametexture(d)', 'istexture(D,Y)']],
#   ['D', ['obj_1(d)', 'obj_2(d)']],
#   ['X', ["ftriangle()", "fcircle()", "fsquare()", "fdiamond()"]],
#   ['Y', ["fplain()", "fcheckered()", "fstripes()", "fdots()"]],
#   ['C', ['True']]
# ]
# test = Mental_grammar(productions, cap=10)

# %% Debug
# x = test.generate_tree()
# x
# d = ((0, 0), (0, 0))
# eval(x[0])

