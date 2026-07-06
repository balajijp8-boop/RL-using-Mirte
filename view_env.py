"""
Interactive viewer: watch the environment with random actions.
Press 'q' in the MuJoCo window to quit.
"""

from mirte_env import MirteStackedBalanceEnv
import numpy as np

env = MirteStackedBalanceEnv(render_mode="human")
obs, info = env.reset(seed=42)

print("Viewer open. Random gentle actions for demo. Press 'q' in the viewer to quit.\n")

for episode in range(5):
    obs, info = env.reset()
    ep_r = 0.0
    for step in range(200):
        # gentle random walk (scaled to 30% to avoid instant drops)
        action = env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        ep_r += reward

        if terminated or truncated:
            outcome = info.get("failure", info.get("success", "timeout"))
            print(f"Episode {episode}, step {step}: {outcome} | reward {ep_r:.1f}")
            break

env.close()
print("Done.")
