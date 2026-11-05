"""Central configuration for the Smart Snake game and its RL environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """Rules of the game (shared by human play, RL env and rule-based agent)."""

    width: int = 12
    height: int = 12
    initial_length: int = 3

    # --- Original additions on top of classic Snake -------------------------
    # Static obstacles generated at every reset (connected board is guaranteed).
    n_obstacles: int = 4
    obstacle_max_size: int = 3  # each obstacle is a short wall segment of 1..max cells
    # Poison apple: eating it shrinks the snake by one segment.
    # It relocates every `poison_ttl` steps, so it acts as a moving hazard.
    n_poison: int = 1
    poison_ttl: int = 40
    min_length: int = 2  # dropping below this length (by poison) kills the snake
    # Hunger: the snake starves if it goes `hunger_limit` steps without eating.
    hunger_limit: int = 150


@dataclass
class RewardConfig:
    """Reward function of the MDP."""

    food: float = 10.0
    poison: float = -5.0
    death: float = -10.0  # wall, obstacle, self-collision, poison-death, starvation
    win: float = 50.0  # board completely filled
    step: float = -0.01  # small time penalty to discourage dawdling
    closer: float = 0.1  # shaping: moved closer to food (Manhattan distance)
    farther: float = -0.1  # shaping: moved away from food


@dataclass
class EnvConfig:
    game: GameConfig = field(default_factory=GameConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Hard safety cap on episode length (truncation, not termination).
    max_episode_steps: int = 5_000
