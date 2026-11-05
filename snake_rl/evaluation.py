"""Shared evaluation loop for the RL agent, the rule-based agent and a random baseline."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .config import EnvConfig
from .env import SnakeEnv
from .game import DeathCause

Policy = Callable[[SnakeEnv, dict], int]


def make_rl_policy(model, deterministic: bool = True) -> Policy:
    def act(env: SnakeEnv, obs: dict) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return act


def make_rule_policy(agent) -> Policy:
    return lambda env, obs: agent.predict(env.game)


def make_random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda env, obs: int(rng.integers(3))


def run_episode(env: SnakeEnv, policy: Policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total_reward, decision_time = 0.0, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = policy(env, obs)
        decision_time += time.perf_counter() - t0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    steps = info["steps"]
    return {
        "seed": seed,
        "score": info["score"],
        "final_length": info["length"],
        "steps": steps,
        "total_reward": total_reward,
        "poison_eaten": info["poison_eaten"],
        "death_cause": info["death_cause"] if terminated else "truncated",
        "won": info["won"],
        "steps_per_apple": steps / info["score"] if info["score"] > 0 else np.nan,
        "decision_ms": 1000 * decision_time / max(steps, 1),
    }


def evaluate(policy: Policy, agent_name: str, n_episodes: int, base_seed: int,
             config: EnvConfig | None = None, progress: bool = True) -> pd.DataFrame:
    env = SnakeEnv(config)  # headless
    rows = []
    it = range(n_episodes)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc=f"eval {agent_name}", unit="ep")
    for i in it:
        row = run_episode(env, policy, seed=base_seed + i)
        row["agent"] = agent_name
        row["episode"] = i
        rows.append(row)
    env.close()
    cols = ["agent", "episode"] + [c for c in rows[0] if c not in ("agent", "episode")]
    return pd.DataFrame(rows)[cols]


def summarize(df: pd.DataFrame, success_threshold: int = 10) -> pd.DataFrame:
    """Aggregate per-episode results into one row of metrics per agent."""
    out = []
    for agent, d in df.groupby("agent", sort=False):
        n = len(d)
        row = {
            "agent": agent,
            "episodes": n,
            "score_mean": d.score.mean(),
            "score_std": d.score.std(ddof=1),
            "score_ci95": 1.96 * d.score.std(ddof=1) / np.sqrt(n),
            "score_median": d.score.median(),
            "score_max": d.score.max(),
            "total_reward_mean": d.total_reward.mean(),
            "total_reward_std": d.total_reward.std(ddof=1),
            "steps_mean": d.steps.mean(),
            "steps_per_apple": d.steps.sum() / max(d.score.sum(), 1),
            f"success_rate_score>={success_threshold}": (d.score >= success_threshold).mean(),
            "poison_per_episode": d.poison_eaten.mean(),
            "win_rate": d.won.mean(),
            "decision_ms": d.decision_ms.mean(),
        }
        for cause in (DeathCause.WALL, DeathCause.OBSTACLE, DeathCause.SELF, DeathCause.POISON,
                      DeathCause.STARVATION, "truncated"):
            row[f"death_{cause}_rate"] = (d.death_cause == cause).mean()
        out.append(row)
    return pd.DataFrame(out)
