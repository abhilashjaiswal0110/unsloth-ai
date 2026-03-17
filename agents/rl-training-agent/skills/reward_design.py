"""
RL Training Agent: Reward Design Skill

Helps design and compose reward functions for GRPO and DPO training.
"""


# Pre-built reward function templates
REWARD_TEMPLATES = {
    "math_correctness": '''
def math_correctness_reward(completions, answers, **kwargs):
    """Reward correct mathematical answers."""
    import re
    scores = []
    for completion, expected in zip(completions, answers):
        score = 0.0
        # Extract boxed answer
        boxed = re.findall(r"\\\\boxed\\{(.+?)\\}", completion)
        if not boxed:
            boxed = re.findall(r"(?:answer|result)\\s*(?:is|=)\\s*([\\d.]+)", completion, re.I)
        if boxed and str(expected).strip() in boxed[-1].strip():
            score += 2.0
        # Reward step-by-step reasoning
        if "step" in completion.lower() or "therefore" in completion.lower():
            score += 0.5
        scores.append(score)
    return scores
''',
    "format_json": '''
def json_format_reward(completions, **kwargs):
    """Reward valid JSON output."""
    import json
    scores = []
    for completion in completions:
        try:
            json.loads(completion)
            scores.append(1.0)
        except json.JSONDecodeError:
            scores.append(-1.0)
    return scores
''',
    "conciseness": '''
def conciseness_reward(completions, **kwargs):
    """Reward concise, focused responses."""
    scores = []
    for completion in completions:
        word_count = len(completion.split())
        if word_count < 50:
            scores.append(1.0)
        elif word_count < 150:
            scores.append(0.5)
        elif word_count < 300:
            scores.append(0.0)
        else:
            scores.append(-0.5)
    return scores
''',
    "safety": '''
def safety_reward(completions, is_safe_prompt=None, **kwargs):
    """Reward safe responses and refusal of harmful requests."""
    REFUSAL_PHRASES = ["i cannot", "i can\\'t", "not appropriate", "sorry, but"]
    scores = []
    for i, completion in enumerate(completions):
        safe = is_safe_prompt[i] if is_safe_prompt else True
        if safe:
            scores.append(1.0 if len(completion) > 20 else 0.0)
        else:
            refused = any(p in completion.lower() for p in REFUSAL_PHRASES)
            scores.append(2.0 if refused else -2.0)
    return scores
''',
    "no_repetition": '''
def no_repetition_reward(completions, **kwargs):
    """Penalize repetitive responses."""
    scores = []
    for completion in completions:
        words = completion.lower().split()
        if len(words) < 5:
            scores.append(0.0)
            continue
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        scores.append(unique_ratio * 2.0 - 1.0)
    return scores
''',
    "step_by_step": '''
def step_by_step_reward(completions, **kwargs):
    """Reward structured, step-by-step reasoning."""
    scores = []
    for completion in completions:
        score = 0.0
        # Check for numbered steps
        import re
        steps = re.findall(r"(?:Step\\s*\\d|\\d\\.\\s)", completion)
        score += min(len(steps) * 0.3, 1.5)
        # Check for conclusion
        if any(w in completion.lower() for w in ["therefore", "thus", "in conclusion", "final answer"]):
            score += 0.5
        scores.append(score)
    return scores
''',
}


def list_available_rewards() -> dict:
    """
    List all available pre-built reward function templates.

    Returns:
        Dictionary mapping reward names to descriptions
    """
    descriptions = {
        "math_correctness": "Rewards correct mathematical answers with step-by-step reasoning",
        "format_json": "Rewards valid JSON output format",
        "conciseness": "Rewards concise, focused responses",
        "safety": "Rewards safe responses and refusal of harmful requests",
        "no_repetition": "Penalizes repetitive text generation",
        "step_by_step": "Rewards structured step-by-step reasoning",
    }
    return descriptions


def get_reward_code(reward_name: str) -> str:
    """
    Get the code for a pre-built reward function.

    Args:
        reward_name: Name of the reward template

    Returns:
        Python code string for the reward function
    """
    return REWARD_TEMPLATES.get(reward_name, f"# Unknown reward: {reward_name}")


def compose_rewards(reward_names: list[str], weights: list[float] = None) -> str:
    """
    Compose multiple reward functions into a single combined reward.

    Args:
        reward_names: List of reward template names to combine
        weights: Optional weights for each reward (equal if None)

    Returns:
        Python code string for the combined reward function
    """
    if weights is None:
        weights = [1.0] * len(reward_names)

    code_parts = []
    func_names = []

    for name in reward_names:
        if name in REWARD_TEMPLATES:
            code_parts.append(REWARD_TEMPLATES[name])
            # Extract function name
            for line in REWARD_TEMPLATES[name].split("\n"):
                if line.startswith("def "):
                    func_name = line.split("(")[0].replace("def ", "")
                    func_names.append(func_name)
                    break

    # Generate combined function
    combined = "\n".join(code_parts)
    combined += f"""

def combined_reward(completions, **kwargs):
    \"\"\"Combined reward from: {', '.join(reward_names)}\"\"\"
    weights = {weights}
    all_scores = []
"""

    for i, func_name in enumerate(func_names):
        combined += f"    scores_{i} = {func_name}(completions, **kwargs)\n"

    combined += "    for j in range(len(completions)):\n"
    combined += "        total = "
    combined += " + ".join(
        f"weights[{i}] * scores_{i}[j]" for i in range(len(func_names))
    )
    combined += "\n"
    combined += "        all_scores.append(total)\n"
    combined += "    return all_scores\n"

    return combined
