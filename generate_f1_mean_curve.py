import json

import matplotlib.pyplot as plt

# 替换为你自己的文件路径
with open("runs/grpo/yy958/qwen3-0.6b-0707/trainer_state.json") as f:
    data = json.load(f)

log_history = data.get("log_history", [])

steps = []
f1_means = []

for entry in log_history:
    if "rewards/reward_f1/mean" in entry:
        steps.append(entry["step"])
        f1_means.append(entry["rewards/reward_f1/mean"])

plt.figure(figsize=(10, 6))
plt.plot(steps, f1_means, marker="o")
plt.title("F1 Reward Mean over Training Steps")
plt.xlabel("Training Step")
plt.ylabel("F1 Reward Mean")
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_reward_curve.png")
plt.show()
