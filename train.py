"""Train the PPO agent on Smart Snake (headless mode).

Usage:
    uv run train.py                         # default: 3M timesteps, 8 parallel envs
    uv run train.py --timesteps 500000 --run-name quick
    uv run tensorboard --logdir logs        # monitor convergence

Outputs:
    models/<run>/final_model.zip     last weights
    models/<run>/best_model.zip      best model on periodic evaluation (EvalCallback)
    models/<run>/checkpoints/        periodic checkpoints
    logs/<run>/                      TensorBoard logs (SB3 + per-episode total reward)
    logs/<run>/episodes.csv          total reward of every training episode
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from snake_rl.callbacks import EpisodeLoggerCallback
from snake_rl.config import EnvConfig
from snake_rl.env import SnakeEnv
from snake_rl.networks import policy_kwargs


def linear_schedule(initial: float):
    return lambda progress_remaining: progress_remaining * initial


def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO agent on Smart Snake")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="ppo_snake")
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (multi-process)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--window", type=int, default=100, help="sliding window for the average reward")
    p.add_argument("--eval-freq", type=int, default=50_000, help="timesteps between evaluations")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    cfg = EnvConfig()
    model_dir = Path("models") / args.run_name
    log_dir = Path("logs") / args.run_name
    if log_dir.exists():
        shutil.rmtree(log_dir)  # start a fresh log for this run name
    model_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(lambda: SnakeEnv(cfg), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    env = VecMonitor(env)  # provides the undiscounted episode return in info["episode"]["r"]
    eval_env = VecMonitor(make_vec_env(lambda: SnakeEnv(cfg), n_envs=1, seed=args.seed + 10_000))

    hyper = dict(
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs(),
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
        **hyper,
    )
    print(model.policy)

    # Save the run configuration for reproducibility / the report
    (model_dir / "config.json").write_text(json.dumps({
        "args": vars(args),
        "env_config": asdict(cfg),
        "hyperparameters": {k: (v if not callable(v) else "linear 3e-4 -> 0") for k, v in hyper.items()},
        "torch": torch.__version__,
        "device": str(model.device),
    }, indent=2))

    per_env = lambda n: max(n // args.n_envs, 1)  # noqa: E731
    callbacks = CallbackList([
        EpisodeLoggerCallback(log_dir, window=args.window),
        EvalCallback(eval_env, n_eval_episodes=20, eval_freq=per_env(args.eval_freq),
                     best_model_save_path=str(model_dir), log_path=str(log_dir / "eval"),
                     deterministic=True),
        CheckpointCallback(save_freq=per_env(500_000), save_path=str(model_dir / "checkpoints"),
                           name_prefix="ppo_snake"),
    ])

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="ppo", progress_bar=True)
    model.save(model_dir / "final_model")
    print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min")
    print(f"Models : {model_dir.resolve()}")
    print(f"Logs   : {log_dir.resolve()}   (uv run tensorboard --logdir logs)")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
