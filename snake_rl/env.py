"""Gymnasium environment wrapping `SnakeGame`.

MDP summary
-----------
* Action space  : Discrete(3), relative to the current heading
                  0 = go straight, 1 = turn right, 2 = turn left.
                  (Relative actions make the reverse/suicide move impossible by design.)
* Observation   : Dict
    - "grid"     : float32 (5, H+2, W+2) image with a 1-cell wall border
                   ch0 head, ch1 body (1.0 at neck -> ~0 at tail, encodes order),
                   ch2 food, ch3 poison, ch4 walls + obstacles
    - "features" : float32 (14,) hand-crafted vector
                   danger straight/right/left (3), heading one-hot (4),
                   food dx, dy (2), nearest poison dx, dy (2),
                   hunger ratio (1), length ratio (1), poison-ahead flag (1)
* Reward        : see `RewardConfig` (food, poison, death, win, step, distance shaping)
* Termination   : death (wall / obstacle / self / poison / starvation) or win.
* Truncation    : `max_episode_steps` safety cap.

render_mode=None  -> headless (pygame is never imported)
render_mode="human" / "rgb_array" -> pygame graphics (snake_rl.renderer)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .game import Direction, Event, SnakeGame

N_CHANNELS = 5
N_FEATURES = 14

ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT = 0, 1, 2
ACTION_NAMES = ["straight", "right", "left"]


def relative_to_absolute(heading: Direction, action: int) -> Direction:
    if action == ACTION_RIGHT:
        return heading.turn_right()
    if action == ACTION_LEFT:
        return heading.turn_left()
    return heading


def absolute_to_relative(heading: Direction, target: Direction) -> int:
    if target == heading.turn_right():
        return ACTION_RIGHT
    if target == heading.turn_left():
        return ACTION_LEFT
    return ACTION_STRAIGHT  # straight (or reverse, which is ignored by the game)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, caption: str = ""):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.caption = caption
        g = self.cfg.game
        self.game = SnakeGame(g, rng=np.random.default_rng())

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(N_CHANNELS, g.height + 2, g.width + 2), dtype=np.float32),
                "features": spaces.Box(-1.0, 1.0, shape=(N_FEATURES,), dtype=np.float32),
            }
        )
        self._renderer = None
        self.total_reward = 0.0

    # ----------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game.reset(rng=self.np_random)
        self.total_reward = 0.0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        g = self.game
        r = self.cfg.reward
        prev_dist = self._food_distance()

        result = g.step(relative_to_absolute(g.direction, int(action)))

        reward = r.step
        if result.event == Event.FOOD:
            reward += r.food
        elif result.event == Event.POISON:
            reward += r.poison
        elif result.event == Event.WIN:
            reward += r.food + r.win
        elif result.event == Event.DEATH:
            reward += r.death
        if result.event in (Event.MOVE, Event.POISON):
            new_dist = self._food_distance()
            if new_dist < prev_dist:
                reward += r.closer
            elif new_dist > prev_dist:
                reward += r.farther

        terminated = result.done
        truncated = (not terminated) and g.steps >= self.cfg.max_episode_steps
        self.total_reward += reward

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info(result.event)

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .renderer import Renderer  # lazy import: headless mode never loads pygame

            self._renderer = Renderer(self.cfg.game, headless=self.render_mode == "rgb_array")
        hud = {"Agent": self.caption} if self.caption else {}
        hud["Reward"] = f"{self.total_reward:.1f}"
        frame = self._renderer.draw(self.game, extra_hud=hud)
        if self.render_mode == "human":
            self._renderer.tick(self.metadata["render_fps"])
            return None
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------- internals
    def _food_distance(self) -> int:
        g = self.game
        if g.food is None:
            return 0
        return abs(g.food[0] - g.head[0]) + abs(g.food[1] - g.head[1])

    def _get_info(self, event: str | None = None) -> dict:
        g = self.game
        return {
            "score": g.score,
            "length": g.length,
            "steps": g.steps,
            "poison_eaten": g.poison_eaten,
            "death_cause": g.death_cause,
            "won": g.won,
            "event": event,
        }

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {"grid": build_grid(self.game), "features": build_features(self.game)}


# ---------------------------------------------------------------------------
# Observation builders (module-level so they can be unit-tested / reused)
# ---------------------------------------------------------------------------
def build_grid(g: SnakeGame) -> np.ndarray:
    W, H = g.cfg.width, g.cfg.height
    grid = np.zeros((N_CHANNELS, H + 2, W + 2), dtype=np.float32)
    # walls (border) + obstacles
    grid[4, 0, :] = grid[4, -1, :] = 1.0
    grid[4, :, 0] = grid[4, :, -1] = 1.0
    for x, y in g.obstacles:
        grid[4, y + 1, x + 1] = 1.0
    n = len(g.snake)
    if g.alive or g.won:
        hx, hy = g.head
        if g.in_bounds(g.head):
            grid[0, hy + 1, hx + 1] = 1.0
    for i, (x, y) in enumerate(g.snake):
        if i == 0:
            continue
        grid[1, y + 1, x + 1] = 1.0 - (i - 1) / n  # neck=1.0, decreasing towards tail
    if g.food is not None:
        grid[2, g.food[1] + 1, g.food[0] + 1] = 1.0
    for x, y in g.poison:
        grid[3, y + 1, x + 1] = 1.0
    return grid


def build_features(g: SnakeGame) -> np.ndarray:
    W, H = g.cfg.width, g.cfg.height
    f = np.zeros(N_FEATURES, dtype=np.float32)
    d = g.direction
    for i, nd in enumerate((d, d.turn_right(), d.turn_left())):
        f[i] = float(g.is_deadly(g.next_pos(nd)))
    f[3 + int(d)] = 1.0
    hx, hy = g.head
    if g.food is not None:
        f[7] = (g.food[0] - hx) / W
        f[8] = (g.food[1] - hy) / H
    if g.poison:
        px, py = min(g.poison, key=lambda p: abs(p[0] - hx) + abs(p[1] - hy))
        f[9] = (px - hx) / W
        f[10] = (py - hy) / H
        f[13] = float(g.next_pos(d) in g.poison)
    f[11] = g.hunger / g.cfg.hunger_limit
    f[12] = g.length / (W * H)
    return np.clip(f, -1.0, 1.0)
