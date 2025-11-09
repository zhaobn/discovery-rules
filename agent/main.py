# %%
import itertools
import random

from openai import OpenAI
from dotenv import load_dotenv
import os

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

    def __init__(self, max_steps=40, combinable_pairs=None, seed=None, history_window=None):
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.combinable_pairs = combinable_pairs or self._generate_combinable_pairs()
        self.history_window = history_window  # show full or partial history
        self.pending_regens = []   # list of tuples: (step_due, shape, texture)
        self.regenerated_ids = set()  # to ensure each base object only regenerates once
        self.reset()

    def _generate_combinable_pairs(self):
        """Hidden logic: randomly choose some combinable pairs (index-based)."""
        all_pairs = list(itertools.product(range(16), range(16)))
        return set(self.random.sample(all_pairs, k=40))  # arbitrary

    def reset(self):
        self.objects = [
            {"id": i, "shape": s, "texture": t, "color": "red"}
            for i, (s, t) in enumerate(itertools.product(self.SHAPES, self.TEXTURES))
        ]
        self.score = 0
        self.steps = 0
        self.history = []
        return self._get_state()

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
                info["error"] = "invalid consume"

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
                info["error"] = "invalid combine"

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
# env = ComposeEnv(seed=42, history_window=10)
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

    def _format_prompt(self, state, available_actions):
        """Create a clean text prompt for the model."""
        action_examples = (
            "Examples of valid actions:\n"
            "- consume 3  (to consume object with id 3)\n"
            "- combine 2 5  (to combine object 2 with object 5)\n"
        )
        return f"""
You are an agent exploring a world of colored objects. 
Your goal is to maximize total points by consuming or combining objects.

Current world state:
{state}

You can take ONE action this step.
Available actions (summarized): {len(available_actions)} possible actions.
{action_examples}

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
        import random
        return random.choice(available_actions)

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
    return pd.DataFrame(records)


# %%
def main():
    load_dotenv()
    env = ComposeEnv(seed=42, history_window=4)
    agent = LLMAgent(model="gpt-4o-mini")
    df = run_episode(env, agent)
    df.to_csv("results.csv", index=False)
    print("Saved results to results.csv")

if __name__ == "__main__":
    main()
# # %%
# load_dotenv()
# # %%
# print("OPENAI_API_KEY loaded:", "OPENAI_API_KEY" in os.environ)
# # %%
# client = OpenAI()
# # %%
# try:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": "Hello, world!"}]
#     )
#     print("Model response:", response.choices[0].message.content.strip())
# except Exception as e:
#     print("API call failed:", e)
# # %%

# %%
