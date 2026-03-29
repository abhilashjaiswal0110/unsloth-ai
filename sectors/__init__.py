# Sector-specific RLHF + GRPO fine-tuning pipeline
# Supports: healthcare, public_utility, insurance
#
# Lazy imports to avoid requiring GPU/torch on import.
# Usage:
#   from sectors.synthetic_data import generate_all_datasets
#   from sectors.reward_functions import make_sector_reward
#   from sectors.evaluate_sector_model import run_evaluation
#   from sectors.train_sector_grpo import train_sector
#   from sectors.run_pipeline import run_pipeline
