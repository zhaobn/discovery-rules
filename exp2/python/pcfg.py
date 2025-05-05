# %%
from random import sample
from math import log

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
#### For evaluation
def fand(x, y): return x and y
def fdisj(x, y): return x or y

def ftriangle(): return 'triangle'
def fcircle(): return 'circle'
def fsquare(): return 'square'
def fdiamond(): return 'diamond'
def fplain(): return 'plain'
def fcheckered(): return 'checkered'
def fstripes(): return 'stripes'
def fdots(): return 'dots'

# Example data ('Egg(S1,O0)', '3', '3')
def stripe(d): return d[0][4:-1].split(',')[0][1:]
def spot(d): return d[0][4:-1].split(',')[1][1:]
def stick(d): return d[1]


# %% Debug
productions = [
  ['S', ['A', 'B', 'fand(A,B)']],
  ['A', ['sameshape(d)', 'diffshape(d)', 'C', 'fdisj(C,C)']],
  ['B', ['sametexture(d)', 'difftexture(d)', 'D', 'fdisj(D,D)']],
  ['C', ['isshape(obj_1(d),X)', 'isshape(obj_2(d),X)', 'fand(isshape(obj_1(d),X),isshape(obj_2(d),X))']],
  ['D', ['istexture(obj_1(d),Y)', 'istexture(obj_2(d),Y)', 'fand(istexture(obj_1(d),Y),istexture(obj_2(d),Y))']],
  ['X', ["ftriangle()", "fcircle()", "fsquare()", "fdiamond()"]],
  ['Y', ["fplain()", "fcheckered()", "fstripes()", "fdots()"]],
]
test = Rational_rules(productions, cap=100)

test.generate_tree()



# # x = test.generate_tree()
# # x = test.generate_tree(logging=False)

# %%
