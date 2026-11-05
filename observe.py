"""Visualise one episode played by the trained RL agent or by the rule-based agent.

Usage:
    uv run observe.py                         # trained PPO agent (models/ppo_snake/best_model.zip)
    uv run observe.py --agent rule            # rule-based agent
    uv run observe.py --agent rl --seed 3 --fps 20   # replay a specific map (default: random seed)
    uv run observe.py --agent rl --gif results/episode.gif   # also save the episode as a GIF
"""

from __future__ import annotations

import argparse

import pygame

from snake_rl.env import SnakeEnv
from snake_rl.rule_agent import RuleBasedAgent


def main():
    p = argparse.ArgumentParser(description="Watch an agent play Smart Snake")
    p.add_argument("--agent", choices=["rl", "rule"], default="rl")
    p.add_argument("--model", default="models/ppo_snake/best_model.zip")
    p.add_argument("--seed", type=int, default=None, help="map/episode seed (default: random)")
    p.add_argument("--fps", type=float, default=12)
    p.add_argument("--gif", default=None, help="optional path to save the episode as an animated GIF")
    args = p.parse_args()

    if args.agent == "rl":
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        act = lambda env, obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
        name = "PPO"
    else:
        agent = RuleBasedAgent()
        act = lambda env, obs: agent.predict(env.game)  # noqa: E731
        name = "Rule-based"

    if args.seed is None:
        # OS entropy: unaffected by PPO.load(), which re-seeds Python's global `random`
        import secrets

        args.seed = secrets.randbelow(1_000_000)
    print(f"Seed {args.seed}  (replay this episode with --seed {args.seed})")

    env = SnakeEnv(render_mode="human", caption=f"{name}  seed {args.seed}")
    env.metadata["render_fps"] = args.fps
    obs, info = env.reset(seed=args.seed)
    frames = []
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                env.close()
                return
        obs, reward, terminated, truncated, info = env.step(act(env, obs))
        done = terminated or truncated
        if args.gif:
            frames.append(env._renderer.draw(env.game, {"Agent": name}))

    r = env._renderer
    msg = r.death_message(env.game) if terminated else "Time limit reached"
    print(f"[{name}] score={info['score']} length={info['length']} steps={info['steps']} "
          f"total_reward={env.total_reward:.2f} end={info['death_cause'] or ('win' if info['won'] else 'truncated')}")
    if args.gif and frames:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
        print(f"GIF saved to {args.gif}")

    # keep the final frame on screen until the window is closed / a key is pressed
    while True:
        r.draw(env.game, {"Agent": name}, overlay=msg, sub_overlay=f"score {info['score']} - press any key")
        ev = pygame.event.wait()
        if ev.type in (pygame.QUIT, pygame.KEYDOWN):
            break
    env.close()


if __name__ == "__main__":
    main()
