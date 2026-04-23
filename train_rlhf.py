"""
RLHF training — Christiano et al. (2017).

"Deep Reinforcement Learning from Human Preferences"
Christiano, Leike, Brown, Martic, Legg, Amodei (NeurIPS 2017)

Baseline RLHF with a single reward model (§2.1).
For §2.2 improvements (ensemble, uncertainty queries, normalisation) see
train_rlhf_ensemble.py.

Two feedback modes:
  oracle (default) — simulated Boltzmann-rational oracle scores clip pairs.
  human  (--human) — real human watches clips and presses A/B/S to label.

Usage
-----
    uv run python train_rlhf.py                          # oracle (automated)
    uv run python train_rlhf.py --human                  # real human labels
    uv run python train_rlhf.py --episodes 200 --seed 0
    uv run python train_rlhf.py --human --episodes 100 --seed 0

Controls (--human mode)
-----------------------
  [A]   — Clip A was better
  [B]   — Clip B was better
  [S]   — Skip (tie / unsure)
  [Esc] — Quit early
"""

import argparse
import collections
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from cartpole import config as cfg
from cartpole.reward_model import RewardModel
from cartpole.train_utils import (
    collect_segment,
    make_agent,
    run_rl_episode,
    sample_preference_pairs,
    save_history_csv,
)

# ---------------------------------------------------------------------------
# Pygame UI constants (used only in --human mode)
# ---------------------------------------------------------------------------

_WIN_W   = 620
_WIN_H   = 460
_FRAME_W = 600
_FRAME_H = 400
_CLIP_FPS = 30

_COL_BG     = (30,  30,  40)
_COL_A      = (100, 160, 240)
_COL_B      = (240, 140,  60)
_COL_SKIP   = (160, 160, 160)
_COL_WHITE  = (255, 255, 255)
_COL_YELLOW = (255, 220,  60)


# ---------------------------------------------------------------------------
# Pygame UI helpers (--human mode only)
# ---------------------------------------------------------------------------

def _init_pygame():
    pygame.init()
    pygame.display.set_caption("RLHF — Human Preference Labelling")
    screen = pygame.display.set_mode((_WIN_W, _WIN_H))
    font_l = pygame.font.SysFont("Arial", 28, bold=True)
    font_s = pygame.font.SysFont("Arial", 18)
    return screen, font_l, font_s


def _blit_frame(screen, frame: np.ndarray) -> None:
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (_FRAME_W, _FRAME_H))
    screen.blit(surf, ((_WIN_W - _FRAME_W) // 2, 0))


def _draw_bar(screen, font, text: str, colour) -> None:
    bar_rect = pygame.Rect(0, _FRAME_H, _WIN_W, _WIN_H - _FRAME_H)
    pygame.draw.rect(screen, _COL_BG, bar_rect)
    label = font.render(text, True, colour)
    screen.blit(label, (_WIN_W // 2 - label.get_width() // 2, _FRAME_H + 14))


def _overlay_label(screen, font, text: str, colour) -> None:
    label = font.render(text, True, colour)
    bg = pygame.Surface((label.get_width() + 20, label.get_height() + 8), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (_WIN_W - bg.get_width()) // 2
    screen.blit(bg, (x, 10))
    screen.blit(label, (x + 10, 14))


def _pump_quit() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _wait_for_keypress(screen, font_s) -> bool:
    _draw_bar(screen, font_s, "Press any key to continue…", _COL_SKIP)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return event.key != pygame.K_ESCAPE


def _play_clip(screen, font_l, font_s, frames, label, colour, clock, fps=_CLIP_FPS) -> bool:
    for frame in frames:
        if _pump_quit():
            return False
        screen.fill(_COL_BG)
        _blit_frame(screen, frame)
        _overlay_label(screen, font_l, label, colour)
        _draw_bar(screen, font_s, f"Watching {label}…  ({fps} fps)", colour)
        pygame.display.flip()
        clock.tick(fps)
    return True


def _query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                 pair_index, total_pairs, fps=_CLIP_FPS) -> float | None:
    """
    Show Clip A then Clip B, ask human which was better.
    Returns 1.0 (A), 0.0 (B), 0.5 (skip), or None (quit).
    """
    header = f"Pair {pair_index}/{total_pairs}"

    screen.fill(_COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP A…", _COL_A)
    pygame.display.flip()
    pygame.time.wait(600)
    if not _play_clip(screen, font_l, font_s, frames_a, "CLIP  A", _COL_A, clock, fps):
        return None

    screen.fill(_COL_BG)
    _blit_frame(screen, frames_a[-1])
    _overlay_label(screen, font_l, "CLIP  A  —  end", _COL_A)
    if not _wait_for_keypress(screen, font_s):
        return None

    screen.fill(_COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP B…", _COL_B)
    pygame.display.flip()
    pygame.time.wait(600)
    if not _play_clip(screen, font_l, font_s, frames_b, "CLIP  B", _COL_B, clock, fps):
        return None

    screen.fill(_COL_BG)
    _blit_frame(screen, frames_b[-1])
    _overlay_label(screen, font_l, "CLIP  B  —  end", _COL_B)
    if not _wait_for_keypress(screen, font_s):
        return None

    # Show thumbnails and wait for A / B / S
    prompt = "Which was better?   [A]  Clip A     [B]  Clip B     [S]  Skip"
    screen.fill(_COL_BG)
    th_w, th_h = _FRAME_W // 2 - 10, _FRAME_H // 2
    for frame, x_off, lbl, col in [
        (frames_a[-1], 5,               "A", _COL_A),
        (frames_b[-1], _FRAME_W // 2 + 5, "B", _COL_B),
    ]:
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (th_w, th_h))
        screen.blit(surf, (x_off, 20))
        tag = font_l.render(f"CLIP {lbl}", True, col)
        screen.blit(tag, (x_off + th_w // 2 - tag.get_width() // 2, th_h + 28))
    _draw_bar(screen, font_s, prompt, _COL_YELLOW)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_a:
                    print(f"  [{pair_index}/{total_pairs}] Human chose: A")
                    return 1.0
                if event.key == pygame.K_b:
                    print(f"  [{pair_index}/{total_pairs}] Human chose: B")
                    return 0.0
                if event.key == pygame.K_s:
                    print(f"  [{pair_index}/{total_pairs}] Human skipped")
                    return 0.5


def _collect_segment_with_frames(
    env: gym.Env,
    agent,
    seg_length: int,
    reward_model=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    obs_list, frames = [], []
    obs, _ = env.reset()
    action = agent.begin_episode(obs)
    while len(obs_list) < seg_length:
        frame = env.render()
        obs_list.append(obs.copy())
        frames.append(frame)
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        reward = reward_model.predict(next_obs) if reward_model is not None else float(env_reward)
        action = agent.act(next_obs, reward)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)
    return np.array(obs_list), frames


def _collect_human_preferences(screen, font_l, font_s, clock, seg_buf, n_pairs, rng, fps=_CLIP_FPS):
    segs_a, segs_b, prefs = [], [], []
    indices = list(range(len(seg_buf)))
    for k in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        obs_a, frames_a = seg_buf[i]
        obs_b, frames_b = seg_buf[j]
        mu = _query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                          pair_index=k + 1, total_pairs=n_pairs, fps=fps)
        if mu is None:
            print("  User quit during labelling.")
            return segs_a, segs_b, prefs
        segs_a.append(obs_a)
        segs_b.append(obs_b)
        prefs.append(mu)
    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Oracle training (automated)
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int, skip_charts: bool = False) -> None:
    warmup_eps   = max(10, int(total_episodes * cfg.RLHF_WARMUP_FRACTION))
    remaining    = total_episodes - warmup_eps
    num_iter     = max(1, remaining // cfg.RLHF_EPISODES_PER_ITER)
    actual_total = warmup_eps + num_iter * cfg.RLHF_EPISODES_PER_ITER

    out = pathlib.Path(cfg.experiment_dir(total_episodes, "rlhf-oracle"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  RLHF (oracle)  —  {actual_total} episodes  seed={seed}")
    print(f"  Warm-up: {warmup_eps} eps  |  Iterations: {num_iter} × {cfg.RLHF_EPISODES_PER_ITER} eps")
    print(f"  Output: {out}")
    print("=" * 60)

    rng   = np.random.default_rng(seed)
    env   = gym.make("CartPole-v1")
    agent = make_agent(rng)
    reward_model = RewardModel(obs_dim=env.observation_space.shape[0], rng=rng)

    seg_buf: collections.deque[np.ndarray] = collections.deque(maxlen=cfg.RLHF_SEGMENT_BUFFER)
    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []

    print("\n=== Phase 1: Warm-up ===")
    for ep in range(warmup_eps):
        episode_lengths.append(run_rl_episode(env, agent))
        if (ep + 1) % max(1, warmup_eps // 5) == 0:
            print(f"  ep {ep+1:4d}  avg(last 10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {cfg.RLHF_WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(cfg.RLHF_WARMUP_SEGMENTS):
        seg_buf.append(collect_segment(env, agent, rng))

    print("Bootstrapping reward model…")
    for _ in range(2):
        segs_a, segs_b, prefs = sample_preference_pairs(list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, rng)
        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
    print(f"  Bootstrap loss: {loss:.4f}")

    print("\n=== Phase 2: RLHF loop ===")
    for it in range(1, num_iter + 1):
        iter_lengths = [
            run_rl_episode(env, agent, reward_model)
            for _ in range(cfg.RLHF_EPISODES_PER_ITER)
        ]
        episode_lengths.extend(iter_lengths)

        for _ in range(cfg.RLHF_SEGMENTS_PER_ITER):
            seg_buf.append(collect_segment(env, agent, rng, reward_model))

        segs_a, segs_b, prefs = sample_preference_pairs(list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, rng)
        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)

        if it % max(1, num_iter // 10) == 0 or it == 1:
            print(f"  iter {it:4d}/{num_iter}"
                  f"  avg_ep={np.mean(iter_lengths):6.1f}"
                  f"  rm_loss={loss:.4f}")

    env.close()

    agent.save(out / f"rlhf_oracle_s{seed}_model.npz")
    reward_model.save(out / f"rlhf_oracle_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"rlhf_oracle_s{seed}_history.csv")
    print(f"\nSaved to {out}/")

    _plot(episode_lengths, rm_losses, warmup_eps,
          f"RLHF (oracle) — {actual_total} eps, seed={seed}",
          "tomato", out / f"rlhf_oracle_s{seed}_results.png",
          skip_charts=skip_charts)


# ---------------------------------------------------------------------------
# Human training (interactive)
# ---------------------------------------------------------------------------

def train_human(total_episodes: int, seed: int) -> None:
    """Train RLHF with real human clip comparisons via pygame UI."""
    warmup_eps  = max(10, int(total_episodes * cfg.RLHF_WARMUP_FRACTION))
    remaining   = total_episodes - warmup_eps
    num_iter    = max(1, remaining // cfg.RLHF_EPISODES_PER_ITER)

    out = pathlib.Path(cfg.experiment_dir(total_episodes, "rlhf-human"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  RLHF (human)  —  {total_episodes} episodes  seed={seed}")
    print(f"  Warm-up: {warmup_eps} eps  |  Iterations: {num_iter} × {cfg.RLHF_EPISODES_PER_ITER} eps")
    print(f"  Output: {out}")
    print("=" * 60)

    rng      = np.random.default_rng(seed)
    env_rgb  = gym.make("CartPole-v1", render_mode="rgb_array")
    env_rl   = gym.make("CartPole-v1")
    agent    = make_agent(rng)
    reward_model = RewardModel(obs_dim=env_rgb.observation_space.shape[0], rng=rng)

    seg_buf: collections.deque[tuple[np.ndarray, list]] = (
        collections.deque(maxlen=cfg.RLHF_SEGMENT_BUFFER)
    )
    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    human_labels     = 0

    screen, font_l, font_s = _init_pygame()
    clock = pygame.time.Clock()

    # Intro screen
    screen.fill(_COL_BG)
    for text, fnt, col, y in [
        ("RLHF  —  Human Preference Labelling", font_l, _COL_YELLOW,  80),
        ("Watch pairs of CartPole clips",        font_s, _COL_WHITE,  150),
        ("Press a key to say which looks better.", font_s, _COL_WHITE, 185),
        ("[A]  Clip A was better",  font_s, _COL_A,    240),
        ("[B]  Clip B was better",  font_s, _COL_B,    275),
        ("[S]  Skip / tie",         font_s, _COL_SKIP,  310),
        ("[Esc]  Quit early",       font_s, _COL_SKIP,  345),
        ("Press any key to start…", font_s, _COL_YELLOW, 410),
    ]:
        surf = fnt.render(text, True, col)
        screen.blit(surf, (_WIN_W // 2 - surf.get_width() // 2, y))
    pygame.display.flip()
    if not _wait_for_keypress(screen, font_s):
        pygame.quit()
        env_rgb.close()
        env_rl.close()
        return

    # Phase 1 — warm-up
    print("=== Phase 1: Warm-up ===")
    for ep in range(warmup_eps):
        episode_lengths.append(run_rl_episode(env_rl, agent))
        if (ep + 1) % max(1, warmup_eps // 5) == 0:
            print(f"  ep {ep+1:3d}  avg(10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {cfg.RLHF_WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(cfg.RLHF_WARMUP_SEGMENTS):
        seg = _collect_segment_with_frames(env_rgb, agent, cfg.RLHF_SEGMENT_LENGTH)
        seg_buf.append(seg)

    # Phase 2 — bootstrap
    print(f"\n=== Phase 2: Bootstrap — {cfg.RLHF_PAIRS_PER_ITER} preference pairs ===")
    segs_a, segs_b, prefs = _collect_human_preferences(
        screen, font_l, font_s, clock, list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, rng
    )
    human_labels += len(prefs)
    if len(prefs) >= 2:
        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
        print(f"  Bootstrap done: {len(prefs)} labels, loss={loss:.4f}")

    # Phase 3 — RLHF loop
    print("\n=== Phase 3: RLHF loop ===")
    for iteration in range(1, num_iter + 1):
        if _pump_quit():
            print("  User quit.")
            break

        iter_lengths = [
            run_rl_episode(env_rl, agent, reward_model)
            for _ in range(cfg.RLHF_EPISODES_PER_ITER)
        ]
        episode_lengths.extend(iter_lengths)

        progress = (iteration - 1) / max(num_iter - 1, 1)
        iter_seg_len = int(25 + 75 * progress)
        iter_fps     = int(15 + 30 * progress)

        for _ in range(cfg.RLHF_SEGMENTS_PER_ITER):
            seg = _collect_segment_with_frames(env_rgb, agent, iter_seg_len, reward_model)
            seg_buf.append(seg)

        segs_a, segs_b, prefs = _collect_human_preferences(
            screen, font_l, font_s, clock, list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, rng, fps=iter_fps
        )
        human_labels += len(prefs)

        if len(prefs) >= 2:
            for _ in range(cfg.RLHF_RM_EPOCHS):
                loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
            rm_losses.append(loss)
        elif rm_losses:
            loss = rm_losses[-1]

        print(f"  Iter {iteration:3d}/{num_iter}"
              f"  avg_ep={np.mean(iter_lengths):6.1f}"
              f"  rm_loss={loss:.4f}"
              f"  labels={human_labels}"
              f"  seg_len={iter_seg_len}")

        if len(prefs) == 0 and cfg.RLHF_PAIRS_PER_ITER > 0:
            break

    env_rgb.close()
    env_rl.close()
    pygame.quit()

    agent.save(out / f"rlhf_human_s{seed}_model.npz")
    reward_model.save(out / f"rlhf_human_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"rlhf_human_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total human labels: {human_labels}")

    _plot(episode_lengths, rm_losses, warmup_eps,
          f"RLHF (human) — {total_episodes} eps, seed={seed}, {human_labels} labels",
          "steelblue", out / f"rlhf_human_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _plot(
    episode_lengths: list[int],
    rm_losses: list[float],
    warmup_eps: int,
    title: str,
    color: str,
    save_path: pathlib.Path,
    skip_charts: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color=color)
    if len(lengths) >= 20:
        rm = np.convolve(lengths, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, len(lengths)), rm, color=color, linewidth=2, label="Rolling mean (20)")
    ax.axvline(warmup_eps, color="orange", linestyle="--", linewidth=1, label="RLHF starts")
    ax.axhline(cfg.GOAL_LENGTH, color="gray", linestyle="--", alpha=0.5,
               label=f"Goal: {cfg.GOAL_LENGTH}")
    ax.set(xlabel="Episode", ylabel="Length", title="Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, cfg.MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color=color, marker="o", markersize=3)
    ax.set(xlabel="Reward model update", ylabel="Preference loss",
           title="Reward model learning")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Plot saved to {save_path}")
    if not skip_charts:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF training (oracle or human preferences)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--human",       action="store_true",
                        help="Use real human clip comparisons instead of simulated oracle")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Save plots to disk but do not display them (for batch runs)")
    args = parser.parse_args()

    if args.human:
        train_human(args.episodes, args.seed)
    else:
        train(args.episodes, args.seed, skip_charts=args.skip_charts)
