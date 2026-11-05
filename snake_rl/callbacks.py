"""Training callback: logs the (undiscounted) total reward of EVERY episode.

Written to:
  * TensorBoard  : episode/total_reward, episode/avg_reward_<W> (sliding window),
                   episode/score, episode/length   (x-axis = episode index)
                   + the same quantities against timesteps under episode_ts/*
  * CSV          : <log_dir>/episodes.csv (used by evaluate.ipynb)
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str | Path, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.window = window
        self.rewards: deque[float] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)
        self.n_episodes = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "episodes"))
        self._csv_file = open(self.log_dir / "episodes.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["episode", "timestep", "total_reward", f"avg_reward_{self.window}",
                            "score", "length", "steps", "death_cause"])

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done or "episode" not in info:
                continue
            self.n_episodes += 1
            ep_r = float(info["episode"]["r"])  # undiscounted return from (Vec)Monitor
            ep_l = int(info["episode"]["l"])
            self.rewards.append(ep_r)
            self.scores.append(info.get("score", 0))
            avg = sum(self.rewards) / len(self.rewards)
            ep, ts = self.n_episodes, self.num_timesteps
            w = self.writer
            w.add_scalar("episode/total_reward", ep_r, ep)
            w.add_scalar(f"episode/avg_reward_{self.window}", avg, ep)
            w.add_scalar("episode/score", info.get("score", 0), ep)
            w.add_scalar("episode/length", ep_l, ep)
            w.add_scalar("episode_ts/total_reward", ep_r, ts)
            w.add_scalar(f"episode_ts/avg_reward_{self.window}", avg, ts)
            w.add_scalar(f"episode_ts/avg_score_{self.window}", sum(self.scores) / len(self.scores), ts)
            self._csv.writerow([ep, ts, ep_r, avg, info.get("score", 0), info.get("length", 0),
                                ep_l, info.get("death_cause")])
        return True

    def _on_rollout_end(self) -> None:
        self._csv_file.flush()
        self.writer.flush()

    def _on_training_end(self) -> None:
        self._csv_file.close()
        self.writer.close()
