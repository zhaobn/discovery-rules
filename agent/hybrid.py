# %%
import itertools
import json
import random

from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime

import re
import time
import pandas as pd
import numpy as np

from agent import ComposeEnv, HypothesisAgent




load_dotenv()
client = OpenAI()

# %%
# prompts
SYSTEM_PROMPT = """
You are a strict semantic parser.

Translate natural language rules about object combinations into a structured JSON DSL.

Allowed primitives:
- same_shape, different_shape
- same_texture, different_texture
- same_color, different_color
- color_is, shape_is, texture_is
- texture_pair (use only for explicit pairs like 'plain with dotted')
- shape_pair (use only for explicit pairs like 'square with circle')

Rules must be:
{
  "type": "rule",
  "valid_if": <logical_expression>
}

Logical expressions:
- {"and": [expr1, expr2, ...]}
- {"or": [expr1, expr2, ...]}
- {"not": expr}
- {"predicate": "<primitive_name>"}
- {"color_is": "<color>"}
- {"shape_is": "<shape>"}
- {"texture_is": "<texture>"}
- {"texture_pair": ["textureA", "textureB"]}
- {"shape_pair": ["shapeA", "shapeB"]}
Return only valid JSON.
"""


FEWSHOT_TEMPLATE = """
Example 1:
Rule: Only objects of the same shape can be combined.
Output:
{{"type": "rule", "valid_if": {{"predicate": "same_shape"}}}}

Example 2:
Rule: Combinations are valid when shapes are the same and textures are different.
Output:
{{"type": "rule", "valid_if": {{"and": [{{"predicate": "same_shape"}}, {{"predicate": "different_texture"}}]}}}}

Example 3:
Rule: Combinations are valid when shapes are the same and textures are plain with dotted.
Output:
{{"type": "rule", "valid_if": {{"and": [{{"predicate": "same_shape"}}, {{"texture_pair": ["plain","dotted"]}}]}}}}

Now parse the following rule(s):
{rules}
"""



# %%
def call_llm_parser(rules, model="gpt-4o-mini"):
    """
    rules: a single string (one rule) or multiple rules separated by newlines
    """
    user_msg = FEWSHOT_TEMPLATE.format(rules=rules)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content


# 
# load_dotenv()
# call_llm_parser("Rule: The combination of shapes is successful when both inputs are of the same type (shape and fill pattern), while combinations of the same type but different fill patterns or colors fail.")
rlt_str = '{"type": "rule", "valid_if": {"and": [{"predicate": "same_shape"}, {"predicate": "same_texture"}]}}'
rlt_obj = json.loads(rlt_str)




# %%
# turn parsed JSON into executable function
ALLOWED_PREDICATES = {
    "same_shape","different_shape","same_texture","different_texture",
    "same_color","different_color"
}
VALID_TEXTURES = {"plain","dotted","striped","checkered"}
VALID_SHAPES = {"square","circle","triangle","diamond"}
VALID_COLORS = {"red","yellow","orange","green","blue","purple"}

def validate_expression(expr):
    if "and" in expr: return all(validate_expression(e) for e in expr["and"])
    if "or" in expr: return all(validate_expression(e) for e in expr["or"])
    if "not" in expr: return validate_expression(expr["not"])
    if "predicate" in expr: return expr["predicate"] in ALLOWED_PREDICATES
    if "color_is" in expr: return expr["color_is"] in VALID_COLORS
    if "shape_is" in expr: return expr["shape_is"] in VALID_SHAPES
    if "texture_is" in expr: return expr["texture_is"] in VALID_TEXTURES
    if "texture_pair" in expr:
        pair = expr["texture_pair"]
        return isinstance(pair, list) and len(pair) == 2 and pair[0] in VALID_TEXTURES and pair[1] in VALID_TEXTURES
    if "shape_pair" in expr:
        pair = expr["shape_pair"]
        return isinstance(pair, list) and len(pair) == 2 and pair[0] in VALID_SHAPES and pair[1] in VALID_SHAPES
    return False

def validate_rule_json(rule_json):
    return rule_json.get("type") == "rule" and "valid_if" in rule_json and validate_expression(rule_json["valid_if"])

# validate_rule_json(rlt_obj)


# %%
# compile the validated JSON into a function
def compile_expression_to_string(expr):
    def expr_to_str(e):
        if "and" in e:
            return "(" + " and ".join(expr_to_str(x) for x in e["and"]) + ")"
        if "or" in e:
            return "(" + " or ".join(expr_to_str(x) for x in e["or"]) + ")"
        if "not" in e:
            return "(not " + expr_to_str(e["not"]) + ")"
        if "predicate" in e:
            p = e["predicate"]
            if p == "same_shape": return "o1['shape'] == o2['shape']"
            if p == "different_shape": return "o1['shape'] != o2['shape']"
            if p == "same_texture": return "o1['texture'] == o2['texture']"
            if p == "different_texture": return "o1['texture'] != o2['texture']"
            if p == "same_color": return "o1['color'] == o2['color']"
            if p == "different_color": return "o1['color'] != o2['color']"
        if "color_is" in e:
            return f'o1["color"] == "{e["color_is"]}" and o2["color"] == "{e["color_is"]}"'
        if "shape_is" in e:
            return f'o1["shape"] == "{e["shape_is"]}" and o2["shape"] == "{e["shape_is"]}"'
        if "texture_is" in e:
            return f'o1["texture"] == "{e["texture_is"]}" and o2["texture"] == "{e["texture_is"]}"'
        if "texture_pair" in e:
            t1, t2 = e["texture_pair"]
            return f'((o1["texture"]=="{t1}" and o2["texture"]=="{t2}") or (o1["texture"]=="{t2}" and o2["texture"]=="{t1}"))'
        raise ValueError("Invalid expression node")

    bool_expr = expr_to_str(expr)
    fn_str = f"""def rule(o1,o2):
    return {bool_expr}
"""
    return fn_str

#compiled_rule = compile_expression_to_string(rlt_obj["valid_if"])



# %%
def generate_transitions_for_env(env, rule_fn_str):
    # Create all 16 base objects (shape × texture)
    norm_objs = [{"shape": s, "texture": t, "color": "red"} 
                for s, t in itertools.product(env.SHAPES, env.TEXTURES)]    
    # Compile the rule function
    local_env = {}
    exec(rule_fn_str, {}, local_env)
    rule_fn = local_env["rule"]
    
    # Generate all unordered unique pairs
    transitions = []
    for o1, o2 in itertools.permutations(norm_objs, 2):
        transitions.append(int(rule_fn(o1, o2)))
    assert len(transitions) == 240
    return transitions, norm_objs

# generate_transitions_for_env(env, compiled_rule)

# %%
def planner_agent(env, hypothesis, n_left):
    """
    Greedy planning agent that:
    1. Parses hypothesis into a Python function via LLM
    2. Generates transitions for shape×texture pairs
    3. Chooses actions using local_greedy_policy
    """
    # --- Parse hypothesis with LLM (mocked here) ---
    llm_json_str = call_llm_parser(hypothesis)  # returns {"valid_if": "...python expression..."}
    try:
        rule_json = json.loads(llm_json_str)
    except:
        raise ValueError("LLM did not return valid JSON")

    # --- Compile to function string ---
    rule_fn_str = compile_expression_to_string(rule_json["valid_if"])  # returns 'def rule(o1,o2): ...'

    # --- Generate transitions and normalized objects ---
    sampled_transitions, norm_objs = generate_transitions_for_env(env, rule_fn_str)

    # --- Build pair_to_idx for 16 norm_objs → 240 unique unordered pairs ---
    pair_to_idx = {}
    for idx, (o1, o2) in enumerate(itertools.permutations(norm_objs, 2)):
        pair_combo = (o1["shape"], o1["texture"], o2["shape"], o2["texture"])
        # print(idx, pair_combo)
        pair_to_idx[pair_combo] = idx

    # --- Convert env objects to normalized indices ---
    available = [(o["shape"], o["texture"], o["color"], o["id"]) for o in env.objects]
    if not available:
        return None 

    # --- Greedy selection ---
    # First, find top-level objects by color points
    points = [env.COLOR_POINTS[o[2]] for o in available]
    if not points:
        return None
    max_points = max(points)
    top_objs = [o for o, p in zip(available, points) if p == max_points]

    # If one action left or max level reached, consume highest
    if n_left == 1 or max_points >= env.COLOR_POINTS["purple"]:
        chosen = top_objs[0]  # can sample randomly
        return ("consume", chosen[3])

    # Try combinations
    np.random.shuffle(top_objs)
    for o_top in top_objs:
        for o_partner in available:
            if o_top[3] == o_partner[3]:
                continue
            key = (o_top[0], o_top[1], o_partner[0], o_partner[1])
            idx = pair_to_idx.get(key)
            if idx is not None and sampled_transitions[idx] == 1:
                return ("combine", o_top[3], o_partner[3])

    # Fallback: consume top
    chosen = top_objs[0]
    return ("consume", chosen[3])


# %%
# env = ComposeEnv(max_steps=10, condition="simple", seed=0)
# env.reset()

# hypothesis = "Objects of the same shape can be combined."
# print("\n=== INITIAL STATE ===")
# print(env._get_state())

# for step in range(5):
#     n_left = env.max_steps - env.steps
#     action = planner_agent(env, hypothesis, n_left)
#     print(f"\nPlanner chose action: {action}")
    
#     state, reward, done, info = env.step(action)
    
#     print("Reward:", reward)
#     print("Info:", info.get("feedback") or info.get("error") or info.get("new_object"))
    
#     if done:
#         break

# print("\n=== FINAL STATE ===")
# print(env._get_state())

# %%
def test_run(
    condition,
    seed,
    run,
    max_steps=40,
    update_every=5,
    save_dir="results",
    file_prefix="exp"
):
    env = ComposeEnv(max_steps=max_steps, condition=condition, seed=seed)
    hypothesis_agent = HypothesisAgent()
    random.seed(seed)

    history_records = []
    hypothesis_records = []
    api_logs = []

    state = env.reset()
    done = False
    hypothesis = None

    SHARED_CONTEXT = """World Rules:
- Object color determines value: red(1), yellow(10), orange(100), green(1K), blue(10K), purple(100K)
- Object features determine combinability: shape (square, circle, triangle, diamond) interacts with texture (plain, dotted, striped, checkered)
- Actions: 'consume X' or 'combine X Y'
- Limited total actions
- Hidden rules determine valid combinations
"""

    while not done:
        step = env.steps
        n_left = max_steps - step

        # Update hypothesis periodically
        if step == 0 or step % update_every == 0:
            hypothesis, prompt = hypothesis_agent.update_hypothesis(SHARED_CONTEXT, env.history)
            hypothesis_records.append({
                "condition": condition, "run": run, "seed": seed,
                "step": step, "hypothesis": hypothesis
            })
            api_logs.append({
                "condition": condition, "run": run, "seed": seed, "step": step,
                "llm_type": "hypothesis",
                "system_message": "Respond only with the requested format.",
                "user_prompt": prompt,
                "model_output_raw": hypothesis,
                "current_score": env.score,
                "object_snapshot": str(env.objects),
            })

        # Plan and execute action
        action = planner_agent(env=env, hypothesis=hypothesis_agent.hypothesis, n_left=n_left)
        if action is None:
            if env.objects:
                action = ("consume", random.choice([o["id"] for o in env.objects]))
            else:
                break # No objects left to act on

        state, reward, done, info = env.step(action)
        args = action[1:]

        history_records.append({
            "condition": condition, "run": run, "seed": seed,
            "step": env.steps,  # after env.step() so it's correct
            "action_type": action[0],
            "action_arg1": args[0] if args else None,
            "action_arg2": args[1] if len(args) > 1 else None,
            "reward": reward,
            "score": env.score,
            "current_hypothesis": hypothesis_agent.hypothesis,
            "feedback": info.get("feedback"),
            "error": info.get("error"),
            "new_object": str(info.get("new_object")) if "new_object" in info else None
        })

    # Final hypothesis
    final_prompt = (
        f"{SHARED_CONTEXT}\n\n"
        f"Full history:\n{env.history}\n\n"
        f"State your best guess about the combination rule in one sentence: 'Rule: [rule]'"
    )
    final_message = hypothesis_agent.client.chat.completions.create(
        model=hypothesis_agent.model,
        messages=[
            {"role": "system", "content": "Respond only with the requested format."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=100
    ).choices[0].message.content.strip()

    hypothesis_records.append({
        "condition": condition, "run": run, "seed": seed,
        "step": env.steps, "hypothesis": final_message
    })
    api_logs.append({
        "condition": condition, "run": run, "seed": seed, "step": env.steps,
        "llm_type": "final_message",
        "user_prompt": final_prompt,
        "model_output_raw": final_message,
        "final_score": env.score
    })

    # Save results
    os.makedirs("results", exist_ok=True)
    run_name = f"{file_prefix}-{condition}_seed{seed}_run{run}"
    dfs = [pd.DataFrame(r) for r in [history_records, hypothesis_records, api_logs]]
    suffixes = ["history", "hypotheses", "api_logs"]
    for df, suffix in zip(dfs, suffixes):
        df.to_csv(f"{save_dir}/{run_name}_{suffix}.csv", index=False)

    print(f"Saved results for {run_name}. Final score: {env.score}")
    return dfs

test_run(condition="simple", seed=0, run=0, max_steps=40, update_every=5, file_prefix="test")

# %%
def run_experiments(
    conditions=["simple", "medium", "hard"],
    seeds=[0, 1, 2, 3, 4],
    runs_per_seed=2,
    max_steps=40,
    update_every=5,
    save_dir="results",
    file_prefix="exp"
):
    os.makedirs(save_dir, exist_ok=True)
    
    all_histories, all_hypotheses, all_logs = [], [], []

    for condition in conditions:
        for seed in seeds:
            for run in range(runs_per_seed):
                print(f"Running condition={condition}, seed={seed}, run={run}...")
                try:
                    df_history, df_hypotheses, df_logs = test_run(
                        condition=condition, seed=seed, run=run,
                        max_steps=max_steps, update_every=update_every,
                        file_prefix=file_prefix, save_dir=save_dir
                    )
                    all_histories.append(df_history)
                    all_hypotheses.append(df_hypotheses)
                    all_logs.append(df_logs)

                except Exception as e:
                    print(f"  Failed: {e}")
                    continue

    # Combine and save
    combined = {
        "history": pd.concat(all_histories, ignore_index=True),
        "hypotheses": pd.concat(all_hypotheses, ignore_index=True),
        "api_logs": pd.concat(all_logs, ignore_index=True),
    }
    for name, df in combined.items():
        df.to_csv(f"{save_dir}/{file_prefix}combined_{name}.csv", index=False)
        print(f"Saved combined {name}: {len(df)} rows")

    return combined["history"], combined["hypotheses"], combined["api_logs"]
# %%
run_experiments(
    conditions=["simple", "medium", "hard"],
    seeds=[0, 1, 2, 3, 4],
    runs_per_seed=2,
    max_steps=40,
    update_every=5,
    save_dir="results",
    file_prefix="hybrid"
)
# %%
