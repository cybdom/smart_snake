"""Pure game logic of Smart Snake (no rendering, no RL code).

The same `SnakeGame` object is used by:
  * the interactive (pygame) mode  -> play.py / observe.py
  * the headless Gymnasium env     -> snake_rl.env.SnakeEnv
  * the rule-based agent           -> snake_rl.rule_agent

Coordinates are (x, y) with x in [0, width) to the right and y in [0, height) downwards.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .config import GameConfig

Pos = tuple[int, int]


class Direction(IntEnum):
    """Absolute directions, ordered clockwise so that +1 = turn right, -1 = turn left."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Pos:
        return _DELTAS[self]

    def turn_right(self) -> "Direction":
        return Direction((self + 1) % 4)

    def turn_left(self) -> "Direction":
        return Direction((self - 1) % 4)

    def opposite(self) -> "Direction":
        return Direction((self + 2) % 4)


_DELTAS: dict[Direction, Pos] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class Event:
    MOVE = "move"
    FOOD = "food"
    POISON = "poison"
    DEATH = "death"
    WIN = "win"


class DeathCause:
    WALL = "wall"
    OBSTACLE = "obstacle"
    SELF = "self"
    POISON = "poison"
    STARVATION = "starvation"


@dataclass
class StepResult:
    event: str
    done: bool
    death_cause: str | None = None


class SnakeGame:
    """Snake on a bounded grid with obstacles, a moving poison apple and hunger."""

    def __init__(self, config: GameConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = config or GameConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        c = self.cfg
        cx, cy = c.width // 2, c.height // 2
        self.direction = Direction.RIGHT
        # head first: snake[0] is the head, snake[-1] the tail
        self.snake: deque[Pos] = deque((cx - i, cy) for i in range(c.initial_length))
        self.body: set[Pos] = set(self.snake)
        self.obstacles: set[Pos] = set()
        self._generate_obstacles()
        self.food: Pos | None = None
        self.poison: list[Pos] = []
        self.score = 0  # number of apples eaten
        self.poison_eaten = 0
        self.steps = 0
        self.hunger = 0  # steps since last apple
        self.poison_timer = 0
        self.alive = True
        self.won = False
        self.death_cause: str | None = None
        self._spawn_food()
        for _ in range(c.n_poison):
            self._spawn_poison()

    def _generate_obstacles(self) -> None:
        """Random short wall segments; the free board must stay a single connected region."""
        c = self.cfg
        head = self.snake[0]
        # Keep the row in front of the snake clear at the start.
        reserved = set(self.snake) | {(head[0] + i, head[1]) for i in range(1, 4)}
        for _attempt in range(100):
            obstacles: set[Pos] = set()
            for _ in range(c.n_obstacles):
                size = int(self.rng.integers(1, c.obstacle_max_size + 1))
                horizontal = bool(self.rng.integers(0, 2))
                x0 = int(self.rng.integers(0, c.width))
                y0 = int(self.rng.integers(0, c.height))
                for k in range(size):
                    p = (x0 + k, y0) if horizontal else (x0, y0 + k)
                    if self.in_bounds(p) and p not in reserved:
                        obstacles.add(p)
            if self._is_connected(obstacles):
                self.obstacles = obstacles
                return
        self.obstacles = set()  # extremely unlikely fallback

    def _is_connected(self, obstacles: set[Pos]) -> bool:
        free = [
            (x, y)
            for x in range(self.cfg.width)
            for y in range(self.cfg.height)
            if (x, y) not in obstacles
        ]
        seen = {free[0]}
        queue = deque([free[0]])
        while queue:
            x, y = queue.popleft()
            for dx, dy in _DELTAS.values():
                n = (x + dx, y + dy)
                if self.in_bounds(n) and n not in obstacles and n not in seen:
                    seen.add(n)
                    queue.append(n)
        return len(seen) == len(free)

    def _free_cells(self) -> list[Pos]:
        blocked = self.body | self.obstacles | set(self.poison)
        if self.food is not None:
            blocked.add(self.food)
        return [
            (x, y)
            for x in range(self.cfg.width)
            for y in range(self.cfg.height)
            if (x, y) not in blocked
        ]

    def _spawn_food(self) -> bool:
        self.food = None
        free = self._free_cells()
        if not free:
            return False
        self.food = free[int(self.rng.integers(len(free)))]
        return True

    def _spawn_poison(self) -> None:
        free = self._free_cells()
        head = self.snake[0]
        # never spawn poison directly next to the head (would be unavoidable/unfair)
        free = [p for p in free if abs(p[0] - head[0]) + abs(p[1] - head[1]) > 1]
        if free:
            self.poison.append(free[int(self.rng.integers(len(free)))])

    # ---------------------------------------------------------------- queries
    @property
    def head(self) -> Pos:
        return self.snake[0]

    @property
    def length(self) -> int:
        return len(self.snake)

    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.cfg.width and 0 <= p[1] < self.cfg.height

    def next_pos(self, direction: Direction, origin: Pos | None = None) -> Pos:
        x, y = origin if origin is not None else self.head
        dx, dy = direction.delta
        return (x + dx, y + dy)

    def collision_cause(self, p: Pos) -> str | None:
        """Why moving the head onto `p` (next step) would be fatal, or None if safe.

        The tail cell counts as free because the tail moves away in the same step
        (unless the snake is about to eat, which cannot happen on its own tail).
        """
        if not self.in_bounds(p):
            return DeathCause.WALL
        if p in self.obstacles:
            return DeathCause.OBSTACLE
        if p in self.body and p != self.snake[-1]:
            return DeathCause.SELF
        return None

    def is_deadly(self, p: Pos) -> bool:
        return self.collision_cause(p) is not None

    # ------------------------------------------------------------------- step
    def step(self, direction: Direction | int | None = None) -> StepResult:
        """Advance the game by one tick moving in absolute `direction`.

        Reversing onto the neck is ignored (the snake keeps going straight).
        """
        if not self.alive:
            return StepResult(Event.DEATH, True, self.death_cause)

        if direction is not None:
            direction = Direction(direction)
            if not (self.length > 1 and direction == self.direction.opposite()):
                self.direction = direction

        self.steps += 1
        self.hunger += 1
        new_head = self.next_pos(self.direction)

        cause = self.collision_cause(new_head)
        if cause is not None:
            return self._die(cause)

        # move head
        self.snake.appendleft(new_head)
        self.body.add(new_head)
        event = Event.MOVE

        if new_head == self.food:
            self.score += 1
            self.hunger = 0
            event = Event.FOOD
            if not self._spawn_food():  # nowhere left to put food -> board is full
                self.won = True
                self.alive = False
                return StepResult(Event.WIN, True)
        else:
            self._pop_tail()
            if new_head in self.poison:
                self.poison.remove(new_head)
                self.poison_eaten += 1
                event = Event.POISON
                if self.length - 1 < self.cfg.min_length:
                    return self._die(DeathCause.POISON)
                self._pop_tail()
                self._spawn_poison()

        # moving poison: relocate periodically
        self.poison_timer += 1
        if self.cfg.poison_ttl > 0 and self.poison_timer >= self.cfg.poison_ttl:
            self.poison_timer = 0
            n = len(self.poison)
            self.poison = []
            for _ in range(n):
                self._spawn_poison()

        if self.hunger >= self.cfg.hunger_limit:
            return self._die(DeathCause.STARVATION)

        return StepResult(event, False)

    def _pop_tail(self) -> None:
        tail = self.snake.pop()
        if tail not in self.snake:  # (only false in degenerate overlapping cases)
            self.body.discard(tail)

    def _die(self, cause: str) -> StepResult:
        self.alive = False
        self.death_cause = cause
        return StepResult(Event.DEATH, True, cause)
