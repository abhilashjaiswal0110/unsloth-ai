# RL Training Agent

An agent for reinforcement learning training workflows with GRPO and DPO using Unsloth.

## Overview

The RL Training Agent guides users through reward-based training:

1. **Reward Design** — Help design effective reward functions
2. **GRPO Setup** — Configure Group Relative Policy Optimization
3. **DPO Setup** — Configure Direct Preference Optimization
4. **Monitoring** — Track reward signals and training stability
5. **Evaluation** — Verify RL-trained model behavior

## Modes

| Mode | Description |
|------|-------------|
| `grpo` | Group Relative Policy Optimization training |
| `dpo` | Direct Preference Optimization training |
| `reward-design` | Help design custom reward functions |
| `evaluate` | Evaluate RL-trained model alignment |

## Usage

### Design a Reward Function

```python
from agents.rl_training_agent.skills.reward_design import design_reward

reward_func = design_reward(
    objectives=["correctness", "conciseness", "safety"],
    task_type="math_reasoning",
)

print(reward_func.code)
```

### Generate GRPO Training Script

```python
from agents.rl_training_agent.skills.rl_training import generate_grpo_script

script = generate_grpo_script(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    dataset_name="gsm8k",
    reward_objectives=["math_correctness", "step_by_step"],
    num_generations=8,
)
```

### AI Assistant Prompt

```
I want to train a model using GRPO to improve its math reasoning.
Help me:
1. Design reward functions for mathematical correctness and clear reasoning
2. Set up GRPO training with Qwen 2.5 7B
3. Configure 8 generations per prompt
4. Add monitoring for reward distribution
5. Evaluate on GSM8K after training
```

## Skills

| Skill | Purpose |
|-------|---------|
| `reward_design` | Design and compose reward functions |
| `rl_training` | Generate GRPO and DPO training scripts |
| `monitor_rl` | Monitor reward signals and training stability |
