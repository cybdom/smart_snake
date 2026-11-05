"""Rule-based baseline agent for Smart Snake.

Decision strategy (evaluated every step, in priority order):

1. **Safe shortest path to the apple.**  BFS from the head to the apple over cells that are
   not walls, obstacles, poison or body.  The path is then *simulated* on a virtual snake;
   it is accepted only if, after eating, the virtual head can still reach the virtual
   tail (otherwise the snake could trap itself in a dead end).
2. **Follow the tail.**  If no safe apple path exists, move along a path towards the tail:
   the tail keeps freeing cells, so this is a survival manoeuvre that buys time until a
   safe apple path opens up.
3. **Maximise free space.**  Otherwise pick the non-fatal move whose resulting region
   (flood fill) is largest, preferring moves that avoid poison.
4. Fallback: go straight (every move is fatal at this point).

The agent reads the full game state (`SnakeGame`) and returns a *relative* action, i.e. the
same action space as the RL agent, so both can be evaluated with the same environment.
"""

from __future__ import annotations

from collections import deque

from .env import absolute_to_relative
from .game import Direction, Pos, SnakeGame

DIRS = list(Direction)


class RuleBasedAgent:
    name = "rule_based"

    def __init__(self, avoid_poison: bool = True):
        self.avoid_poison = avoid_poison

    # public API compatible with SB3's `model.predict`
    def predict(self, game: SnakeGame) -> int:
        target = self.choose_direction(game)
        return absolute_to_relative(game.direction, target)

    # ---------------------------------------------------------------- policy
    def choose_direction(self, g: SnakeGame) -> Direction:
        legal = [d for d in DIRS if d != g.direction.opposite()]
        poison = set(g.poison) if self.avoid_poison else set()

        # 1) safe shortest path to food
        if g.food is not None:
            path = self._bfs(g, g.head, g.food, blocked_extra=poison)
            if path and self._path_is_safe(g, path):
                return self._dir_to(g.head, path[0])

        # 2) follow the tail
        tail_dir = self._tail_chase(g, poison)
        if tail_dir is not None:
            return tail_dir

        # 3) maximise reachable area
        best, best_score = None, -1.0
        for d in legal:
            p = g.next_pos(d)
            if g.is_deadly(p):
                continue
            area = self._flood_area(g, p)
            score = area - (0.5 * area if p in poison else 0)  # discourage poison
            if score > best_score:
                best, best_score = d, score
        return best if best is not None else g.direction

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _dir_to(a: Pos, b: Pos) -> Direction:
        dx, dy = b[0] - a[0], b[1] - a[1]
        for d in DIRS:
            if d.delta == (dx, dy):
                return d
        raise ValueError(f"{a} and {b} are not adjacent")

    @staticmethod
    def _neighbors(g: SnakeGame, p: Pos):
        for d in DIRS:
            n = (p[0] + d.delta[0], p[1] + d.delta[1])
            if g.in_bounds(n) and n not in g.obstacles:
                yield n

    def _bfs(self, g: SnakeGame, start: Pos, goal: Pos, blocked_extra: set[Pos] | None = None,
             body: set[Pos] | None = None, allow: Pos | None = None) -> list[Pos] | None:
        """Shortest path start->goal (exclusive of start). Body cells are blocked except the tail."""
        body = g.body if body is None else body
        blocked = (body - {g.snake[-1]}) | (blocked_extra or set())
        if allow is not None:
            blocked.discard(allow)
        blocked.discard(goal)
        prev: dict[Pos, Pos | None] = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for n in self._neighbors(g, cur):
                if n not in prev and n not in blocked:
                    prev[n] = cur
                    q.append(n)
        if goal not in prev:
            return None
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def _path_is_safe(self, g: SnakeGame, path: list[Pos]) -> bool:
        """Simulate following `path`; check the head can still reach the tail afterwards."""
        snake = deque(g.snake)
        for i, p in enumerate(path):
            snake.appendleft(p)
            if i < len(path) - 1:  # last cell is the apple -> grow (no tail pop)
                snake.pop()
        if len(snake) >= g.cfg.width * g.cfg.height - len(g.obstacles):
            return True
        head, tail = snake[0], snake[-1]
        blocked = set(snake) - {tail}
        return self._reachable(g, head, tail, blocked)

    def _reachable(self, g: SnakeGame, start: Pos, goal: Pos, blocked: set[Pos]) -> bool:
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self._neighbors(g, cur):
                if n == goal:
                    return True
                if n not in seen and n not in blocked:
                    seen.add(n)
                    q.append(n)
        return False

    def _tail_chase(self, g: SnakeGame, poison: set[Pos]) -> Direction | None:
        """Move towards the tail, preferring the move that keeps it reachable and is farthest away
        (taking the long way round leaves more room for the body to clear)."""
        if g.length < 3:
            return None
        tail = g.snake[-1]
        best, best_dist = None, -1
        for d in DIRS:
            if d == g.direction.opposite():
                continue
            p = g.next_pos(d)
            if g.is_deadly(p) or p in poison or p == g.food:
                continue
            # after moving to p the tail moves to snake[-2]
            new_body = (set(g.snake) | {p}) - {tail}
            new_tail = g.snake[-2]
            if not self._reachable(g, p, new_tail, new_body - {new_tail}):
                continue
            dist = abs(p[0] - new_tail[0]) + abs(p[1] - new_tail[1])
            if dist > best_dist:
                best, best_dist = d, dist
        return best

    def _flood_area(self, g: SnakeGame, start: Pos) -> int:
        blocked = g.body - {g.snake[-1]}
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self._neighbors(g, cur):
                if n not in seen and n not in blocked:
                    seen.add(n)
                    q.append(n)
        return len(seen)
