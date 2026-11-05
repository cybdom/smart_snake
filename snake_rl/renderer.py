"""Pygame renderer used by the interactive mode (play.py, observe.py, env render)."""

from __future__ import annotations

import os

import numpy as np

from .config import GameConfig
from .game import DeathCause, Direction, SnakeGame

CELL = 36
HUD_H = 64

BG = (24, 26, 33)
GRID_A = (32, 35, 44)
GRID_B = (36, 39, 49)
SNAKE_HEAD = (120, 220, 120)
SNAKE_BODY = (70, 175, 90)
SNAKE_TAIL = (40, 110, 60)
FOOD = (235, 75, 75)
POISON = (170, 90, 230)
OBSTACLE = (110, 115, 130)
TEXT = (230, 232, 238)
MUTED = (150, 155, 170)
HUNGER_OK = (90, 200, 120)
HUNGER_LOW = (230, 90, 80)

DEATH_TEXT = {
    DeathCause.WALL: "Hit the wall",
    DeathCause.OBSTACLE: "Crashed into an obstacle",
    DeathCause.SELF: "Bit its own tail",
    DeathCause.POISON: "Poisoned",
    DeathCause.STARVATION: "Starved",
}


class Renderer:
    def __init__(self, cfg: GameConfig, headless: bool = False, title: str = "Smart Snake"):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        pygame.init()
        self.cfg = cfg
        self.size = (cfg.width * CELL, cfg.height * CELL + HUD_H)
        if headless:
            self.screen = pygame.Surface(self.size)
        else:
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption(title)
        self.headless = headless
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 18, bold=True)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 14)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 30, bold=True)

    # ------------------------------------------------------------------ draw
    def draw(self, g: SnakeGame, extra_hud: dict | None = None, overlay: str | None = None,
             sub_overlay: str | None = None) -> np.ndarray:
        pg = self.pg
        s = self.screen
        s.fill(BG)
        self._draw_hud(g, extra_hud or {})

        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                pg.draw.rect(s, GRID_A if (x + y) % 2 else GRID_B, self._rect(x, y))

        for x, y in g.obstacles:
            pg.draw.rect(s, OBSTACLE, self._rect(x, y), border_radius=4)
            pg.draw.rect(s, (80, 84, 96), self._rect(x, y).inflate(-12, -12), border_radius=2)

        if g.food is not None:
            pg.draw.circle(s, FOOD, self._center(*g.food), CELL // 2 - 5)
            cx, cy = self._center(*g.food)
            pg.draw.line(s, (90, 160, 80), (cx, cy - CELL // 2 + 6), (cx + 4, cy - CELL // 2 + 2), 3)

        for p in g.poison:
            cx, cy = self._center(*p)
            pg.draw.circle(s, POISON, (cx, cy), CELL // 2 - 5)
            # small "x" marks it as dangerous
            k = 5
            pg.draw.line(s, (40, 20, 60), (cx - k, cy - k), (cx + k, cy + k), 3)
            pg.draw.line(s, (40, 20, 60), (cx - k, cy + k), (cx + k, cy - k), 3)

        n = len(g.snake)
        for i, (x, y) in enumerate(reversed(g.snake)):
            if not g.in_bounds((x, y)):
                continue
            t = i / max(n - 1, 1)
            col = tuple(int(SNAKE_TAIL[c] + (SNAKE_BODY[c] - SNAKE_TAIL[c]) * t) for c in range(3))
            if i == n - 1:
                col = SNAKE_HEAD
            pg.draw.rect(s, col, self._rect(x, y).inflate(-4, -4), border_radius=8)
        if g.in_bounds(g.head):
            self._draw_eyes(g.head, g.direction)

        if overlay:
            shade = pg.Surface((self.size[0], self.size[1] - HUD_H), pg.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            s.blit(shade, (0, HUD_H))
            self._text_center(overlay, self.big, TEXT, HUD_H + (self.size[1] - HUD_H) // 2 - 20)
            if sub_overlay:
                self._text_center(sub_overlay, self.small, MUTED, HUD_H + (self.size[1] - HUD_H) // 2 + 20)

        if not self.headless:
            pg.display.flip()
            pg.event.pump()
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2))

    def _draw_hud(self, g: SnakeGame, extra: dict) -> None:
        pg = self.pg
        s = self.screen
        pg.draw.rect(s, (18, 19, 24), (0, 0, self.size[0], HUD_H))
        left = f"Score {g.score}   Len {g.length}   Steps {g.steps}"
        s.blit(self.font.render(left, True, TEXT), (10, 8))
        extra_txt = "   ".join(f"{k}: {v}" for k, v in extra.items())
        if extra_txt:
            s.blit(self.small.render(extra_txt, True, MUTED), (10, 40))
        # hunger / energy bar: thin full-width strip at the bottom of the HUD
        frac = max(0.0, 1.0 - g.hunger / g.cfg.hunger_limit)
        y0, bar_h = HUD_H - 6, 4
        pg.draw.rect(s, (60, 62, 72), (0, y0, self.size[0], bar_h))
        pg.draw.rect(s, HUNGER_OK if frac > 0.3 else HUNGER_LOW, (0, y0, int(self.size[0] * frac), bar_h))

    def _draw_eyes(self, head, direction: Direction) -> None:
        cx, cy = self._center(*head)
        dx, dy = direction.delta
        px, py = -dy, dx  # perpendicular
        for side in (-1, 1):
            ex = cx + dx * 6 + px * 7 * side
            ey = cy + dy * 6 + py * 7 * side
            self.pg.draw.circle(self.screen, (250, 250, 250), (ex, ey), 4)
            self.pg.draw.circle(self.screen, (20, 20, 20), (ex + dx * 1, ey + dy * 1), 2)

    def _text_center(self, txt, font, col, y) -> None:
        surf = font.render(txt, True, col)
        self.screen.blit(surf, ((self.size[0] - surf.get_width()) // 2, y))

    def _rect(self, x, y):
        return self.pg.Rect(x * CELL, HUD_H + y * CELL, CELL, CELL)

    def _center(self, x, y):
        return (x * CELL + CELL // 2, HUD_H + y * CELL + CELL // 2)

    # ------------------------------------------------------------- utilities
    def death_message(self, g: SnakeGame) -> str:
        if g.won:
            return "Board cleared - you win!"
        return DEATH_TEXT.get(g.death_cause, "Game over")

    def tick(self, fps: float) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        self.pg.display.quit()
        self.pg.quit()
