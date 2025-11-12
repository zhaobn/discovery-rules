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

    def __init__(self, max_steps=40, condition='simple', seed=None, history_window=None):
        self.max_steps = max_steps
        self.condition = condition
        self.random = random.Random(seed)
        self.history_window = history_window  # show full or partial history
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
                msg = f"Step {self.steps}: Consumed {obj['color']} {obj['shape']} {obj['texture']} (id {obj_id}) for {reward} points."
                self._log(msg)
            else:
                msg = f"Step {self.steps}: Invalid consume attempt on id {obj_id}."
                self._log(msg)
                info["feedback"] = "invalid consume"

        elif action[0] == "combine":
            id1, id2 = action[1], action[2]
            pair = (id1, id2)
            o1 = next((o for o in self.objects if o["id"] == id1), None)
            o2 = next((o for o in self.objects if o["id"] == id2), None)

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
                    f"Step {self.steps}: Combined id {id1} ({o1['color']} {o1['shape']} {o1['texture']}) "
                    f"+ id {id2} ({o2['color']} {o2['shape']} {o2['texture']}) → "
                    f"new {new_color} {new_obj['shape']} {new_obj['texture']} (id {new_obj['id']})."
                )
                self._log(msg)
                info["new_object"] = new_obj
            else:
                msg = f"Step {self.steps}: Failed to combine id {id1} and id {id2}."
                self._log(msg)
                info["feedback"] = "invalid combine"

        else:
            msg = f"Step {self.steps}: Unknown action {action}."
            self._log(msg)
            info["error"] = "unknown action"

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Return both world state and truncated history."""
        desc = ", ".join(
            [f"{o['id']}:{o['color']} {o['shape']} {o['texture']}" for o in self.objects]
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
# env = ComposeEnv(seed=42, condition='simple')
# state = env.reset()
# print(state)

# state, reward, done, info = env.step(("consume", 0))
# print(state)

# state, reward, done, info = env.step(("combine", 1, 2))
# print(state)

# state, reward, done, info = env.step(("combine", 4, 2))
# print(state)

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

    def update_hypothesis(self, history):
        """Update hypothesis based on the full history"""
        prompt = (
            "You are trying to infer the hidden rule for combining objects.\n"
            f"History so far:\n{history}\n\n"
            "Write a brief statement about your current hypothesis for which object pairs can combine."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        self.hypothesis = response.choices[0].message.content.strip()
        return self.hypothesis

# %%
class PlannerAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def choose_next_actions(self, env_summary, available_actions, hypothesis, steps_to_plan=1):
        """
        Decide next steps based on hypothesis.
        Returns a list of actions [(action_type, ...), ...]
        """
        prompt = (
            f"Current hypothesis: {hypothesis}\n"
            f"Environment:\n{env_summary}\n"
            f"Available actions:\n{available_actions}\n"
            f"Plan the next {steps_to_plan} actions to maximize points."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
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
    def __init__(self, env, hypothesis_agent, planner_agent, save_dir="results"):
        self.env = env
        self.hypothesis_agent = hypothesis_agent
        self.planner_agent = planner_agent
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Records
        self.history_records = []
        self.hypothesis_records = []
        self.api_logs = []

    def run_episode(self, update_hypothesis_every=5, plan_ahead=1, run_name="run1"):
        state = self.env.reset()
        done = False

        while not done:
            step = self.env.steps
            env_summary = self.env._get_state()
            available_actions = self.env.available_actions()

            # 1️⃣ Update hypothesis if it's time
            if step == 0 or step % update_hypothesis_every == 0:
                hypo_prompt = str(self.env.history)
                hypothesis = self.hypothesis_agent.update_hypothesis(self.env.history)
                self.hypothesis_records.append({
                    "step": step,
                    "hypothesis": hypothesis
                })
                self.api_logs.append({
                    "step": step,
                    "llm_type": "hypothesis",
                    "prompt": hypo_prompt,
                    "model_output": hypothesis
                })

            # 2️⃣ Plan next actions
            plan_prompt = f"Current hypothesis: {self.hypothesis_agent.hypothesis}\nEnv summary:\n{env_summary}\nActions: {available_actions}"
            actions = self.planner_agent.choose_next_actions(
                env_summary, available_actions, self.hypothesis_agent.hypothesis, steps_to_plan=plan_ahead
            )
            self.api_logs.append({
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
                "step": step,
                "action": action,
                "reward": reward,
                "score": self.env.score,
                "info": info,
                "hypothesis": self.hypothesis_agent.hypothesis
            })

        # 5️⃣ Final message to next player
        message_prompt = (
            "You finished the episode. Based on your final hypothesis and history, "
            "write a helpful message for the next player."
        )
        message = self.hypothesis_agent.client.chat.completions.create(
            model=self.hypothesis_agent.model,
            messages=[{"role": "user", "content": message_prompt}],
        ).choices[0].message.content.strip()
        self.history_records.append({
            "step": self.env.steps + 1,
            "action": "message_to_next_player",
            "reward": 0,
            "score": self.env.score,
            "info": message,
            "hypothesis": self.hypothesis_agent.hypothesis
        })
        self.hypothesis_records.append({
            "step": self.env.steps + 1,
            "hypothesis": message
        })
        self.api_logs.append({
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
env = ComposeEnv(seed=42, condition="medium", max_steps=2)
hypo_agent = HypothesisAgent(model="gpt-4o-mini")
planner_agent = PlannerAgent(model="gpt-4o-mini")
executor = ExecutorAgent(env, hypo_agent, planner_agent, save_dir="results")

# Run one episode
df_hist, df_hypo, df_logs = executor.run_episode(update_hypothesis_every=5, plan_ahead=2, run_name="run1_medium")

# %%
