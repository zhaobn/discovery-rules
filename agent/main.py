# %%
import itertools
import random

from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime

import re
import time
import pandas as pd


# %% 
# Environment
class ComposeEnv:
    COLORS = ["red", "yellow", "orange", "green", "blue", "purple"]
    COLOR_POINTS = {"red": 1, "yellow": 10, "orange": 100, "green": 1000, "blue": 10000, "purple": 100000}
    SHAPES = ["square", "circle", "triangle", "diamond"]
    TEXTURES = ["plain", "dotted", "striped", "checkered"]

    def __init__(self, max_steps=40, condition='simple', seed=None, history_window=None, use_full_features=False):
        self.max_steps = max_steps
        self.condition = condition
        self.random = random.Random(seed)
        self.history_window = history_window  # show full or partial history
        self.use_full_features = use_full_features # use object id for full features
        self.pending_regens = []   # list of tuples: (step_due, shape, texture)
        self.regenerated_ids = set()  # to ensure each base object only regenerates once
        self.reset()

    def reset(self):
        self.objects = [
            {"id": i, "shape": s, "texture": t, "color": "red"}
            for i, (s, t) in enumerate(itertools.product(self.SHAPES, self.TEXTURES))
        ]
        self.score = 0
        self.steps = 0
        self.history = []
        self._generate_combinable_pairs(self.condition)
        return self._get_state()

    def _generate_combinable_pairs(self, condition):
        all_pairs = list(itertools.combinations(self.objects, 2))

        def shape_idx(o): return self.SHAPES.index(o["shape"])
        def texture_idx(o): return self.TEXTURES.index(o["texture"])

        if condition == "simple":
            rule_fn = lambda o1, o2: o1["shape"] == o2["shape"]
        elif condition == "medium":
            rule_fn = lambda o1, o2: (shape_idx(o1) + shape_idx(o2) == 3 
                                      and o1["texture"] != o2["texture"])
        elif condition == "hard":
            rule_fn = lambda o1, o2: (shape_idx(o1) + shape_idx(o2) == 3 
                                      and (texture_idx(o1) % 2 == 0 or texture_idx(o2) % 2 == 0))
        else:
            raise ValueError(f"Unknown condition: {condition}")

        self.combinable_pairs = [(o1["id"], o2["id"]) for o1, o2 in all_pairs if rule_fn(o1, o2)]


    def _color_up(self, c1, c2):
        """Returns the next color level based on the higher of the two."""
        i = max(self.COLORS.index(c1), self.COLORS.index(c2))
        return self.COLORS[min(i + 1, len(self.COLORS) - 1)]

    def _log(self, msg):
        self.history.append(msg)

    def step(self, action):
        """Action format: ('consume', id) or ('combine', id1, id2)"""
        self.steps += 1

        # Handle scheduled regenerations
        new_regens = [r for r in self.pending_regens if r[0] == self.steps]
        for _, shape, texture in new_regens:
            new_id = max(o["id"] for o in self.objects) + 1 if self.objects else 0
            self.objects.append({"id": new_id, "shape": shape, "texture": texture, "color": "red"})

        # Remove completed regenerations
        self.pending_regens = [r for r in self.pending_regens if r[0] > self.steps]

        done = self.steps >= self.max_steps
        reward = 0
        info = {}

        if action[0] == "consume":
            obj_id = action[1]
            obj = next((o for o in self.objects if o["id"] == obj_id), None)
            if obj:
                reward = self.COLOR_POINTS[obj["color"]]
                self.score += reward
                self.objects.remove(obj)
                # Schedule regeneration if it was a base red object and not regenerated before
                if obj["color"] == "red" and obj["id"] not in self.regenerated_ids:
                    regen_step = self.steps + 2
                    self.pending_regens.append((regen_step, obj["shape"], obj["texture"]))
                    self.regenerated_ids.add(obj["id"])
                # Logging
                msg = f"Step {self.steps}: Consumed id {obj_id} ({obj['color']} {obj['shape']} {obj['texture']}) for {reward} points."
                self._log(msg)
                info["feedback"] = msg
            else:
                msg = f"Step {self.steps}: Invalid consume attempt on id {obj_id}."
                self._log(msg)
                info["error"] = msg

        elif action[0] == "combine":
            id1, id2 = action[1], action[2]
            pair = (id1, id2)
            o1 = next((o for o in self.objects if o["id"] == id1), None)
            o2 = next((o for o in self.objects if o["id"] == id2), None)

            o1_desc = f"{o1['color']} {o1['shape']} {o1['texture']}" if o1 else "None"
            o2_desc = f"{o2['color']} {o2['shape']} {o2['texture']}" if o2 else "None"

            if o1 and o2 and pair in self.combinable_pairs:
                new_color = self._color_up(o1["color"], o2["color"])
                new_obj = {
                    "id": max(o["id"] for o in self.objects) + 1,
                    "shape": o1["shape"],
                    "texture": o2["texture"],
                    "color": new_color,
                }
                self.objects.remove(o1)
                self.objects.remove(o2)
                self.objects.append(new_obj)
                msg = (
                    f"Step {self.steps}: Combined id {id1} ({o1_desc}) + id {id2} ({o2_desc}) → "
                    f"new {new_color} {new_obj['shape']} {new_obj['texture']} (id {new_obj['id']})."
                )
                self._log(msg)
                info["new_object"] = new_obj
            else:
                msg = f"Step {self.steps}: Failed to combine id {id1} ({o1_desc}) and id {id2} ({o2_desc})."
                self._log(msg)
                info["feedback"] = msg

        else:
            msg = f"Step {self.steps}: Unknown action {action}."
            self._log(msg)
            info["error"] = msg

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Return both world state and truncated history."""
        desc = ", ".join(
            [f"id {o['id']} ({o['color']} {o['shape']} {o['texture']})" for o in self.objects]
        )
        hist = self.history[-self.history_window:] if self.history_window else self.history
        history_str = "\n".join(hist) if hist else "No prior actions."
        return f"Score: {self.score}. Step: {self.steps}/{self.max_steps}.\n" \
               f"Objects: {desc}.\n\nHistory:\n{history_str}"

    def available_actions(self):
        """List all possible consume and combine actions."""
        ids = [o["id"] for o in self.objects]
        consume_actions = [("consume", i) for i in ids]
        combine_actions = [("combine", i, j) for i in ids for j in ids if i != j]
        return consume_actions + combine_actions

# %%
# Test environment
env = ComposeEnv(seed=42, condition='simple')
state = env.reset()
print(state)

state, reward, done, info = env.step(("consume", 0))
print(state)

state, reward, done, info = env.step(("combine", 1, 2))
print(state)

state, reward, done, info = env.step(("combine", 4, 2))
print(state)

# %%
class LLMAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0, max_retries=3):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def _format_prompt(self, state):
        """Create a clean text prompt for the model."""
        action_examples = (
            "Examples of valid actions:\n"
            "- consume 3  (to consume object with id 3)\n"
            "- combine 2 5  (to combine object 2 with object 5)\n"
        )
        return f"""
You are an agent exploring a world of colored objects. 
You can either consume an object to gain points based on its color - red (1 point), yellow (10), orange (100), green (1,000), blue (10,000), purple (100,000) - 
or combine two objects to create a new object of a higher color level.
Some hidden rules determine which objects can be combined.
Try to discover these rules by experimenting with combinations,
that way, you can create higher-level colored objects that yield more points when consumed.
Your goal is to maximize total points within 40 actions.

Current world state:
{state}

{action_examples}

You can take ONE action this step.

Respond with only one line:
Action: <your choice here>
"""

    def _parse_action(self, text):
        """Extract ('consume', id) or ('combine', id1, id2) from model output."""
        m = re.search(r"consume\s+(\d+)", text, re.IGNORECASE)
        if m:
            return ("consume", int(m.group(1)))

        m = re.search(r"combine\s+(\d+)\s+(\d+)", text, re.IGNORECASE)
        if m:
            return ("combine", int(m.group(1)), int(m.group(2)))

        return None  # fallback if nothing valid found

    def choose_action(self, env_state, available_actions):
        """Query the model to decide on an action."""
        prompt = self._format_prompt(env_state, available_actions)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                action = self._parse_action(text)
                if action:
                    return action
                else:
                    print(f"[warn] Invalid LLM response: {text}")
            except Exception as e:
                print(f"[error] API call failed (attempt {attempt+1}): {e}")
                time.sleep(1)

        # fallback to a random action if all retries fail
        return random.choice(available_actions)

    def review_and_write_message(self, history, final_score):
        """Ask the LLM to summarize what it learned and leave a message."""
        prompt = (
            "You have just finished playing a grid-world game where you could "
            "consume and combine objects to earn points.\n\n"
            f"Here is your full history:\n{history}\n\n"
            f"Your total points: {final_score}\n\n"
            "Write a short, helpful message for the next player explaining what "
            "you think are the most important strategies or patterns to do well. "
            "Be clear and concise."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful agent leaving advice for the next player."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# %%
def run_episode(env, agent):
    state = env.reset()
    done = False
    records = []

    while not done:
        actions = env.available_actions()
        action = agent.choose_action(state, actions)
        state, reward, done, info = env.step(action)

        records.append({
            "step": env.steps,
            "action": action,
            "reward": reward,
            "score": env.score,
            "info": info,
        })
    
    message = agent.review_and_write_message(env.history, env.score)

    # Convert message into a final DataFrame row
    final_row = {
        "step": len(env.history) + 1,
        "action": "message_to_next_player",
        "reward": 0,
        "score": env.score,
        "info": {"message": message},
    }
    df = pd.DataFrame(records)
    df = pd.concat([df, pd.DataFrame([final_row])], ignore_index=True)
    return df


# # %%
# def main(save_dir="results", base_name="", steps=40):
#     load_dotenv()

#     os.makedirs(save_dir, exist_ok=True)

#     # Format timestamp like 2025-11-12_14-30-59
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     conditions = ["simple", "medium", "hard"]
#     agent = LLMAgent(model="gpt-4o-mini")

#     for cond in conditions:
#         print(f"\n=== Running condition: {cond} ===")
#         env = ComposeEnv(seed=42, condition=cond, max_steps=steps)
#         df = run_episode(env, agent)

#         filename = f"{base_name}_{cond}_{timestamp}.csv"
#         save_path = os.path.join(save_dir, filename)

#         df.to_csv(save_path, index=False)
#         print(f"✅ Saved results to {save_path}")


# if __name__ == "__main__":
#    main(base_name="default")


# %%
# new attempts with hypothesis testing
class HypothesisAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()
        self.hypothesis = "No hypothesis yet."

    def update_hypothesis(self, SHARED_CONTEXT, history):
        """Update hypothesis based on the full history"""
        prompt = (
            f"{SHARED_CONTEXT}\n\n"
            f"History so far:\n{history}\n\n"
            f"State your best guess about the combination rule in one sentence: 'Rule: [rule]'"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Respond only with the requested format."},
                {"role": "user", "content": prompt}],
                max_tokens=100
        )

        self.hypothesis = response.choices[0].message.content.strip()
        return self.hypothesis

# %%
class PlannerAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def choose_next_actions(self, SHARED_CONTEXT, env_summary, available_actions, hypothesis, steps_to_plan=1):
        """
        Decide next steps based on hypothesis.
        Returns a list of actions [(action_type, ...), ...]
        """
        prompt = (
            f"{SHARED_CONTEXT}\n\n"
            f"Current hypothesis: {hypothesis}\n\n"
            f"Environment state:\n{env_summary}\n\n"

            f"Plan the next actions to maxize points, considering:\n"
            "- Test the hypothesis if uncertain\n"
            "- Build higher-value objects through combinations when rules are known\n"
            "- Consume high-value objects near the action limit\n"
            "- Balance exploration (testing rules) vs exploitation (maximizing points)\n\n"
    
            "You can take ONE action this step.\n"
            f"Available actions:\n{available_actions}\n\n"
            "Respond with only one line:\n"
            "Action: <your choice here>"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Respond only with the requested format."},
                {"role": "user", "content": prompt}]
        )
        # parse into action tuples (assume _parse_actions is implemented)
        return self._parse_actions(response.choices[0].message.content)

    def _parse_actions(self, text):
        """Convert model output into a list of action tuples"""
        actions = []
        import re
        for line in text.splitlines():
            m = re.match(r"consume\s+(\d+)", line, re.IGNORECASE)
            if m:
                actions.append(("consume", int(m.group(1))))
                continue
            m = re.match(r"combine\s+(\d+)\s+(\d+)", line, re.IGNORECASE)
            if m:
                actions.append(("combine", int(m.group(1)), int(m.group(2))))
        return actions

# %%
class ExecutorAgent:
    def __init__(self, env, hypothesis_agent, planner_agent, 
                 condition, run_id, seed, save_dir="results"):
        self.env = env
        self.hypothesis_agent = hypothesis_agent
        self.planner_agent = planner_agent

        self.condition = condition
        self.run_id = run_id
        self.seed = seed

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Records
        self.history_records = []
        self.hypothesis_records = []
        self.api_logs = []

    def run_episode(self, update_hypothesis_every=5, plan_ahead=1, run_name="run1"):
        state = self.env.reset()
        done = False

        SHARED_CONTEXT = """World Rules:
- Object color determines value: red(1), yellow(10), orange(100), green(1K), blue(10K), purple(100K)
- Object features determine combinability: shape (square, circle, triangle, diamond) interacts with texture (plain, dotted, striped, checkered)
- Actions: 'consume X' (gain points) or 'combine X Y' (create new, more valuable object if valid)
- 40 actions total to maximize points
- Hidden rules determine valid combinations
"""

        while not done:
            step = self.env.steps
            env_summary = self.env._get_state()
            available_actions = self.env.available_actions()

            # 1️⃣ Update hypothesis if it's time
            if step == 0 or step % update_hypothesis_every == 0:
                hypo_prompt = str(self.env.history)
                hypothesis = self.hypothesis_agent.update_hypothesis(SHARED_CONTEXT, self.env.history)
                self.hypothesis_records.append({
                    "condition": self.condition,
                    "run": self.run_id,
                    "seed": self.seed,
                    "step": step,
                    "hypothesis": hypothesis
                })
                self.api_logs.append({
                    "condition": self.condition,
                    "run": self.run_id,
                    "seed": self.seed,
                    "step": step,
                    "llm_type": "hypothesis",
                    "prompt": SHARED_CONTEXT + hypo_prompt,
                    "model_output": hypothesis
                })

            # 2️⃣ Plan next actions
            plan_prompt = f"Current hypothesis: {self.hypothesis_agent.hypothesis}\nEnv summary:\n{env_summary}\nActions: {available_actions}"
            actions = self.planner_agent.choose_next_actions(
                SHARED_CONTEXT,
                env_summary, available_actions, self.hypothesis_agent.hypothesis, 
                steps_to_plan=plan_ahead
            )
            self.api_logs.append({
                "condition": self.condition,
                "run": self.run_id,
                "seed": self.seed,
                "step": step,
                "llm_type": "planner",
                "prompt": plan_prompt,
                "model_output": str(actions)
            })

            # 3️⃣ Execute first planned action
            action = actions[0] if actions else random.choice(available_actions)
            state, reward, done, info = self.env.step(action)

            # 4️⃣ Record step
            self.history_records.append({
                "condition": self.condition,
                "run": self.run_id,
                "seed": self.seed,
                "step": step,
                "action": action,
                "reward": reward,
                "score": self.env.score,
                "info": info,
            })

        # 5️⃣ Final message to next player
        message_prompt = (
            f"History:\n{self.env.history}\n\nState your best guess about the combination rule in one sentence: 'Rule: [rule]'"
        )

        message = self.hypothesis_agent.client.chat.completions.create(
            model=self.hypothesis_agent.model,
            messages=[
                {"role": "system", "content": "Respond only with the requested format."},
                {"role": "user", "content": message_prompt}
            ],
            max_tokens=100 # Limit token count to force brevity
          ).choices[0].message.content.strip()  

        self.hypothesis_records.append({
            "condition": self.condition,
            "run": self.run_id,
            "seed": self.seed,
            "step": self.env.steps + 1,
            "hypothesis": message
        })
        self.api_logs.append({
            "condition": self.condition,
            "run": self.run_id,
            "seed": self.seed,
            "step": self.env.steps + 1,
            "llm_type": "message",
            "prompt": message_prompt,
            "model_output": message
        })

        # 6️⃣ Save CSVs
        df_history = pd.DataFrame(self.history_records)
        df_hypothesis = pd.DataFrame(self.hypothesis_records)
        df_logs = pd.DataFrame(self.api_logs)

        df_history.to_csv(os.path.join(self.save_dir, f"{run_name}_history.csv"), index=False)
        df_hypothesis.to_csv(os.path.join(self.save_dir, f"{run_name}_hypotheses.csv"), index=False)
        df_logs.to_csv(os.path.join(self.save_dir, f"{run_name}_api_logs.csv"), index=False)

        print(f"Saved: {run_name}_history.csv, _hypotheses.csv, _api_logs.csv")
        return df_history, df_hypothesis, df_logs


# %%
load_dotenv()
condition = "medium"
run = 4
seed = 42
env = ComposeEnv(seed=42, condition=condition, max_steps=2)
hypo_agent = HypothesisAgent(model="gpt-4o-mini")
planner_agent = PlannerAgent(model="gpt-4o-mini")
executor = ExecutorAgent(
    env,
    hypo_agent,
    planner_agent,
    condition=condition,
    run_id=run,
    seed=seed,
    save_dir="results"
)
# df_hist, df_hypo, df_logs = executor.run_episode(
#     update_hypothesis_every=1,
#     plan_ahead=1,
#     run_name=f"test_{condition}_run{run}"
# )
# %%
def run_experiments(
    conditions=["simple", "medium", "hard"],
    seeds=[0, 1, 2],
    runs_per_seed=1,
    save_dir="results",
    update_every=5,
    plan_ahead=1,
    file_prefix="t1_"
):
    os.makedirs(save_dir, exist_ok=True)

    all_history = []
    all_hypotheses = []
    all_logs = []

    for condition in conditions:
        for seed in seeds:
            for run in range(runs_per_seed):

                print(f"\n=== Running condition={condition}, seed={seed}, run={run} ===")

                env = ComposeEnv(seed=seed, condition=condition)
                hypo_agent = HypothesisAgent(model="gpt-4o-mini")
                planner_agent = PlannerAgent(model="gpt-4o-mini")

                executor = ExecutorAgent(
                    env,
                    hypo_agent,
                    planner_agent,
                    save_dir=save_dir,
                    condition=condition,
                    run_id=run,
                    seed=seed
                )

                df_hist, df_hypo, df_log = executor.run_episode(
                    update_hypothesis_every=update_every,
                    plan_ahead=plan_ahead
                )

                # Append to global
                all_history.append(df_hist)
                all_hypotheses.append(df_hypo)
                all_logs.append(df_log)

    # Save combined results
    pd.concat(all_history).to_csv(
        os.path.join(save_dir, f"{file_prefix}history.csv"), index=False
    )
    pd.concat(all_hypotheses).to_csv(
        os.path.join(save_dir, f"{file_prefix}hypotheses.csv"), index=False
    )
    pd.concat(all_logs).to_csv(
        os.path.join(save_dir, f"{file_prefix}api_logs.csv"), index=False
    )

    print("\nAll experiments complete! Saved:")
    print(f"- {save_dir}/{file_prefix}history.csv")
    print(f"- {save_dir}/{file_prefix}hypotheses.csv")
    print(f"- {save_dir}/{file_prefix}api_logs.csv")



# %%
run_experiments(
    conditions=["simple", 'medium', 'hard'],
    seeds=[0, 1, 2, 3, 4],
    runs_per_seed=2,
    save_dir="results",
    update_every=5,
    plan_ahead=1,
    file_prefix="full_dumb_"
)
# %%
