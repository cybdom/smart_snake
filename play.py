"""Interactive mode: a human plays Smart Snake.

Controls:  arrow keys / WASD = move   P = pause   R = restart   Esc = quit
The game speeds up as the score increases.

Usage:
    uv run play.py
    uv run play.py --seed 7 --fps 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from snake_rl.config import GameConfig
from snake_rl.game import Direction, SnakeGame
from snake_rl.renderer import Renderer

KEYS = {
    pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
    pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
}


def main():
    p = argparse.ArgumentParser(description="Play Smart Snake")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=float, default=7.0, help="starting speed (moves per second)")
    args = p.parse_args()

    cfg = GameConfig()
    rng = np.random.default_rng(args.seed)
    game = SnakeGame(cfg, rng=rng)
    ui = Renderer(cfg, title="Smart Snake - human player")
    best = 0
    paused = False
    started = False
    queue: list[Direction] = []  # buffered turns so quick key combos are not lost

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p and game.alive:
                    paused = not paused
                elif event.key == pygame.K_r or (not game.alive and event.key in (pygame.K_SPACE, pygame.K_RETURN)):
                    game.reset()
                    queue.clear()
                    paused, started = False, False
                elif event.key in KEYS and game.alive:
                    started = True
                    if len(queue) < 3:
                        queue.append(KEYS[event.key])

        if game.alive and started and not paused:
            direction = None
            while queue:  # skip buffered keys that would be ignored (same / reverse)
                d = queue.pop(0)
                last = game.direction
                if d != last and d != last.opposite():
                    direction = d
                    break
            game.step(direction)
            best = max(best, game.score)

        hud = {"Best": best, "Keys": "arrows/WASD  P pause  R restart"}
        if not started and game.alive:
            ui.draw(game, hud, overlay="Smart Snake", sub_overlay="press an arrow key to start")
        elif paused:
            ui.draw(game, hud, overlay="Paused", sub_overlay="press P to resume")
        elif not game.alive:
            ui.draw(game, hud, overlay=ui.death_message(game),
                    sub_overlay=f"score {game.score} - press R / Space to restart")
        else:
            ui.draw(game, hud)
        speed = args.fps + 0.4 * game.score if (game.alive and started and not paused) else 30
        ui.tick(min(speed, 20))

    ui.close()


if __name__ == "__main__":
    main()
