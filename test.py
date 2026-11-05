"""Evaluate the trained RL agent and the rule-based agent (headless) and save CSV results.

Both agents play the SAME set of seeded episodes (identical obstacle layouts and apple
sequences as long as the trajectories coincide), so the comparison is paired and fair.

Usage:
    uv run test.py                                    # 200 episodes each, best model
    uv run test.py --model models/ppo_snake/final_model.zip --episodes 500
    uv run test.py --no-random                        # skip the random baseline

Outputs:
    results/eval_episodes.csv    one row per (agent, episode)
    results/eval_summary.csv     aggregated metrics per agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from snake_rl.evaluation import evaluate, make_random_policy, make_rl_policy, make_rule_policy, summarize
from snake_rl.rule_agent import RuleBasedAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RL vs rule-based agent")
    p.add_argument("--model", default="models/ppo_snake/best_model.zip")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1_000_000, help="base seed (unseen during training)")
    p.add_argument("--out", default="results")
    p.add_argument("--stochastic", action="store_true", help="sample RL actions instead of argmax")
    p.add_argument("--no-random", action="store_true", help="do not evaluate the random baseline")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    model_path = Path(args.model)
    if model_path.exists():
        model = PPO.load(model_path, device="cpu")
        frames.append(evaluate(make_rl_policy(model, deterministic=not args.stochastic), "ppo",
                               args.episodes, args.seed))
    else:
        print(f"[warning] model not found at {model_path} - run train.py first. Skipping RL agent.")
    frames.append(evaluate(make_rule_policy(RuleBasedAgent()), "rule_based", args.episodes, args.seed))
    if not args.no_random:
        frames.append(evaluate(make_random_policy(args.seed), "random", args.episodes, args.seed))

    episodes = pd.concat(frames, ignore_index=True)
    summary = summarize(episodes)
    episodes.to_csv(out / "eval_episodes.csv", index=False)
    summary.to_csv(out / "eval_summary.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["agent", "score_mean", "score_std", "score_max", "total_reward_mean", "steps_per_apple",
            "success_rate_score>=10", "death_self_rate", "death_starvation_rate", "decision_ms"]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print(f"\nSaved {out / 'eval_episodes.csv'} and {out / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
