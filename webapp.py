"""
CartPole HCRL Web Visualizer
=============================
Watch trained CartPole models play directly in your browser.

Usage:
    uv run python webapp.py
    Open: http://localhost:5000
"""

import base64
import io
import json
import pathlib
import re
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context
from PIL import Image

from cartpole.agents import QLearningAgent

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True
RESULTS_DIR = pathlib.Path("experiment-results")
EP_FILTER = "ep200"  # Only show results from this episode count

# Define experiments for the "Report" view
EXPERIMENTS = [
    {
        "id": "baseline_vs_hcrl",
        "title": "Thí nghiệm 1: Baseline vs HCRL Full",
        "subtitle": "Khả năng tăng tốc học tập của con người",
        "description": "So sánh Q-Learning thuần thúy (Baseline) với phương pháp HCRL nhận phản hồi từ Oracle và Human. Mục tiêu là chứng minh sự can thiệp của con người giúp Agent ổn định nhanh hơn.",
        "models": ["baseline_s0", "full_feedback_s0", "hcrl_human_s0"],
        "chart_types": ["training_curves", "box_plot"],
    },
    {
        "id": "timing",
        "title": "Thí nghiệm 2: Tác động của Thời điểm (Timing)",
        "subtitle": "Can thiệp sớm hay muộn thì hiệu quả hơn?",
        "description": "Chúng ta chia giai đoạn huấn luyện thành các cửa sổ: Early (0-20%), Mid (40-60%) và Late (80-100%). Thí nghiệm tìm ra 'Golden Window' để can thiệp hiệu quả nhất.",
        "models": ["early_s0", "mid_s0", "late_s0"],
        "chart_types": ["convergence", "success_rate"],
    },
    {
        "id": "weight_sensitivity",
        "title": "Thí nghiệm 3: Độ nhạy của Trọng số (Weight Sensitivity)",
        "subtitle": "Trọng số feedback bao nhiêu là tối ưu?",
        "description": "So sánh tác động của giá trị phần thưởng từ con người (Feedback Weight) ở các mức 5, 20, và 50. Thí nghiệm này kiểm tra xem việc tăng cường độ tín hiệu khen/chê có giúp model học nhanh hơn không.",
        "models": ["fw5/hcrl_oracle_s0", "fw20/hcrl_oracle_s0", "fw50/hcrl_oracle_s0"],
        "chart_types": ["training_curves", "box_plot"],
    }
]


@app.after_request
def _ngrok_headers(response):
    # Skip ngrok's browser-warning interstitial page
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_NAME_MAP = {
    "baseline":      "Baseline",
    "early":         "Early (0-20%)",
    "mid":           "Mid (40-60%)",
    "late":          "Late (80-100%)",
    "full_feedback": "Full Feedback",
    "hcrl":          "HCRL (interactive)",
}


def make_label(npz: pathlib.Path) -> str:
    """Human-readable label derived from a model file path."""
    try:
        rel = npz.relative_to(RESULTS_DIR)
    except ValueError:
        rel = npz

    ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "")
    ep_tag = f" ({ep})" if ep else ""
    stem = re.sub(r"_model$", "", npz.stem)

    # w20_s1  →  Weight=20 s1 (ep200)
    m = re.match(r"w(\d+)_s(\d+)$", stem)
    if m:
        return f"Weight={m.group(1)} s{m.group(2)}{ep_tag}"

    # Handle fw in parent directory (e.g. hcrl-oracle-fw20)
    fw_match = re.search(r"fw(\d+)", str(npz))
    seed_match = re.search(r"_s(\d+)", stem)
    if fw_match:
        s_tag = f" s{seed_match.group(1)}" if seed_match else ""
        return f"Weight={fw_match.group(1)}{s_tag}{ep_tag}"

    # early_s0  →  Early (0-20%) s0 (ep200)
    m = re.match(r"(early|mid|late|full_feedback)_s(\d+)$", stem)
    if m:
        return f"{_NAME_MAP[m.group(1)]} s{m.group(2)}{ep_tag}"

    # baseline_s2  →  Baseline s2 (ep200)
    m = re.match(r"baseline_s(\d+)$", stem)
    if m:
        return f"Baseline s{m.group(1)}{ep_tag}"

    return _NAME_MAP.get(stem, stem.replace("_", " ").title()) + ep_tag


_REWARD_MODEL_PATTERNS = re.compile(
    r"(^|_)(reward_model|hcrl_reward_model)(\.npz)?$", re.IGNORECASE
)


def _is_agent_model(npz: pathlib.Path) -> bool:
    """Return True only if the .npz file is a QLearningAgent (has q_table key)."""
    if _REWARD_MODEL_PATTERNS.search(npz.stem):
        return False
    try:
        with np.load(npz) as data:
            return "q_table" in data
    except Exception:
        return False


def scan_models() -> list[dict]:
    """Recursively find all QLearningAgent .npz files and return structured metadata."""
    if not RESULTS_DIR.exists():
        return []
    models = []
    for npz in sorted(RESULTS_DIR.rglob("*.npz")):
        if not _is_agent_model(npz):
            continue
        rel = npz.relative_to(RESULTS_DIR)
        ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "misc")

        # Only show models matching our filter
        if ep != EP_FILTER:
            continue

        category = npz.parent.name if npz.parent.name != RESULTS_DIR.name else "root"
        models.append({
            "path":     str(npz).replace("\\", "/"),
            "label":    make_label(npz),
            "ep":       ep,
            "category": category,
            "group":    f"{ep} / {category}",
        })
    return models


# ---------------------------------------------------------------------------
# Gameplay streaming
# ---------------------------------------------------------------------------

def _encode_frame(env: gym.Env, max_w: int = 480, quality: int = 82) -> str:
    """Render one frame → base64 JPEG string."""
    frame = env.render()
    h, w = frame.shape[:2]
    if w > max_w:
        new_w = max_w
        new_h = int(h * max_w / w)
        img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
    else:
        img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_gameplay(model_paths: list[str], num_episodes: int, fps: int):
    """
    Generator that drives all selected models forward in lock-step and yields
    SSE events containing base64-encoded frames + live stats.
    """
    agents, envs, labels = [], [], []

    for path in model_paths:
        p = pathlib.Path(path)
        if not _is_agent_model(p):
            raise ValueError(
                f"{p.name} is not a QLearningAgent model (missing q_table). "
                "Reward model .npz files cannot be played in the web visualizer."
            )
        agents.append(QLearningAgent.load(p))
        envs.append(gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=500))
        labels.append(make_label(p))

    n = len(model_paths)
    history = [[] for _ in range(n)]
    frame_dt = 1.0 / fps
    # Scale frames down when many models are shown simultaneously
    max_w = 480 if n <= 2 else (360 if n <= 4 else 280)

    try:
        for ep in range(num_episodes):
            observations, actions = [], []
            # Use a fixed seed for each episode index so all models face the SAME starting state
            ep_seed = 1000 + ep 
            for agent, env in zip(agents, envs):
                obs, _ = env.reset(seed=ep_seed)
                observations.append(obs)
                actions.append(agent.begin_episode(obs))

            dones = [False] * n
            steps = [0] * n

            while not all(dones):
                t0 = time.perf_counter()

                for i in range(n):
                    if not dones[i]:
                        obs, _, term, trunc, _ = envs[i].step(actions[i])
                        observations[i] = obs
                        steps[i] += 1
                        if term or trunc:
                            dones[i] = True
                            history[i].append(steps[i])
                        else:
                            actions[i] = agents[i].act(obs, reward=0.0)

                frames = [_encode_frame(env, max_w=max_w) for env in envs]
                stats = [
                    {
                        "label":     labels[i],
                        "episode":   ep + 1,
                        "steps":     steps[i],
                        "done":      dones[i],
                        "mean":      round(float(np.mean(history[i])), 1) if history[i] else 0,
                        "best":      int(max(history[i])) if history[i] else 0,
                        "completed": len(history[i]),
                    }
                    for i in range(n)
                ]
                yield _sse({
                    "type":    "frame",
                    "episode": ep + 1,
                    "total":   num_episodes,
                    "frames":  frames,
                    "stats":   stats,
                })

                spare = frame_dt - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)

            time.sleep(0.4)  # brief pause between episodes

        # Final summary sent once after all episodes complete
        summary = [
            {
                "label":     labels[i],
                "mean":      round(float(np.mean(history[i])), 1) if history[i] else 0,
                "median":    round(float(np.median(history[i])), 1) if history[i] else 0,
                "best":      int(max(history[i])) if history[i] else 0,
                "worst":     int(min(history[i])) if history[i] else 0,
                "goal_rate": round(
                    sum(1 for x in history[i] if x >= 195) / len(history[i]) * 100, 1
                ) if history[i] else 0,
                "history":   history[i],
            }
            for i in range(n)
        ]
        yield _sse({"type": "done", "summary": summary})

    except GeneratorExit:
        pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return _HTML


@app.route("/logo")
def logo():
    """Serve the HUST logo from the project root if present."""
    import mimetypes
    from flask import send_file, abort
    for name in ("hust_logo.png", "hust_logo.jpg", "hust_logo.jpeg", "hust_logo.svg"):
        p = pathlib.Path(name)
        if p.exists():
            mime = mimetypes.guess_type(name)[0] or "image/png"
            return send_file(p, mimetype=mime)
    abort(404)


@app.route("/api/models")
def api_models():
    return jsonify(scan_models())


@app.route("/api/experiments")
def api_experiments():
    # Enrich experiments with real model paths from filesystem
    models = scan_models()
    enriched = []
    for exp in EXPERIMENTS:
        exp_models = []
        for m_key in exp["models"]:
            # Match by stem name or partial match
            found = next((m for m in models if m_key in m["path"].lower()), None)
            if found:
                exp_models.append(found)
        
        # Also find relevant CSVs for charts
        csvs = scan_csvs()
        exp_csvs = []
        for m_key in exp["models"]:
            found_csv = next((c for c in csvs if m_key in c["path"].lower()), None)
            if found_csv:
                exp_csvs.append(found_csv["path"])

        copy = exp.copy()
        copy["actual_models"] = exp_models
        copy["actual_csvs"] = exp_csvs
        enriched.append(copy)
    return jsonify(enriched)


@app.route("/api/play")
def api_play():
    paths    = request.args.getlist("models")
    episodes = max(1, min(int(request.args.get("episodes", 5)), 50))
    fps      = max(5, min(int(request.args.get("fps", 30)), 60))

    if not paths:
        return jsonify({"error": "No models selected"}), 400

    @stream_with_context
    def generate():
        yield from stream_gameplay(paths, episodes, fps)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# CSV discovery for charts
# ---------------------------------------------------------------------------

_HISTORY_PATTERN = re.compile(r"_history\.csv$", re.IGNORECASE)


def _csv_label(csv_path: pathlib.Path) -> str:
    """Human-readable label from a history CSV path."""
    try:
        rel = csv_path.relative_to(RESULTS_DIR)
    except ValueError:
        rel = csv_path
    stem = csv_path.stem.replace("_history", "").replace("_episode", "")
    ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "")
    ep_tag = f" ({ep})" if ep else ""
    nice = _NAME_MAP.get(stem, stem.replace("_", " ").title())
    parent = csv_path.parent.name
    if parent not in ("experiment-results", ep.replace("ep", "ep")) and parent != RESULTS_DIR.name:
        nice = f"{parent}/{nice}"
    return nice + ep_tag


def _csv_family(csv_path: pathlib.Path) -> str:
    """Return a seed-stripped family key for grouping multi-seed CSVs."""
    stem = csv_path.stem  # e.g. baseline_s0_history
    # Strip seed suffix: "_s0_history" → "_history"
    base = re.sub(r"_s\d+", "", stem)  # e.g. baseline_history
    parent = csv_path.parent
    try:
        rel_parent = parent.relative_to(RESULTS_DIR)
    except ValueError:
        rel_parent = parent
    return str(rel_parent / base).replace("\\", "/")


def scan_csvs() -> list[dict]:
    """Find all *_history.csv files and return metadata."""
    if not RESULTS_DIR.exists():
        return []
    csvs = []
    for p in sorted(RESULTS_DIR.rglob("*_history.csv")):
        try:
            df = pd.read_csv(p, nrows=2)
            if "episode_length" not in df.columns:
                continue
        except Exception:
            continue
        rel = p.relative_to(RESULTS_DIR)
        ep = next((part for part in rel.parts if re.match(r"ep\d+$", part)), "misc")

        # Only show history files matching our filter
        if ep != EP_FILTER:
            continue

        category = p.parent.name if p.parent.name != RESULTS_DIR.name else "root"
        csvs.append({
            "path":     str(p).replace("\\", "/"),
            "label":    _csv_label(p),
            "ep":       ep,
            "category": category,
            "group":    f"{ep} / {category}",
            "family":   _csv_family(p),
        })
    return csvs


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

_CHART_COLORS = [
    "#4361ee", "#e63946", "#2dc653", "#f77f00", "#7209b7",
    "#3a86a7", "#fb5607", "#8338ec", "#06d6a0", "#ef476f",
    "#118ab2", "#ffd166", "#073b4c", "#ff006e", "#8ac926",
]


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _load_history(path_str: str) -> pd.DataFrame | None:
    p = pathlib.Path(path_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "episode_length" not in df.columns:
            return None
        return df
    except Exception:
        return None


# Style registry: map family-key *patterns* to display properties.
# Patterns are matched against the seed-stripped filename stem
# (e.g. "baseline_history", "early_history", "rlhf_oracle_history").
_MODEL_STYLES: list[tuple[str, dict]] = [
    (r"baseline",           {"label": "Baseline",              "color": "blue",       "linestyle": "--"}),
    (r"early",              {"label": "HCRL Early (0-20%)",    "color": "green",      "linestyle": "-"}),
    (r"mid",                {"label": "HCRL Mid (40-60%)",     "color": "orange",     "linestyle": "-"}),
    (r"late",               {"label": "HCRL Late (80-100%)",   "color": "purple",     "linestyle": "-"}),
    (r"full_feedback",      {"label": "HCRL Full Feedback",    "color": "red",        "linestyle": "-"}),
    (r"hcrl_oracle",        {"label": "HCRL Oracle",           "color": "darkgreen",  "linestyle": "-"}),
    (r"hcrl_human",         {"label": "HCRL Human",            "color": "limegreen",  "linestyle": "-."}),
    (r"fw5",                {"label": "Weight = 5",            "color": "cyan",       "linestyle": "-"}),
    (r"fw20",               {"label": "Weight = 20",           "color": "magenta",    "linestyle": "-"}),
    (r"fw50",               {"label": "Weight = 50",           "color": "gold",       "linestyle": "-"}),
    (r"rlhf_oracle",        {"label": "RLHF Oracle",           "color": "darkorange", "linestyle": "-"}),
    (r"rlhf_ensemble",      {"label": "RLHF Ensemble",         "color": "brown",      "linestyle": "-"}),
    (r"rlhf_human",         {"label": "RLHF Human",            "color": "salmon",     "linestyle": "-."}),
    (r"vi_tamer_human",     {"label": "VI-TAMER Human",        "color": "teal",       "linestyle": "-."}),
    (r"vi_tamer",           {"label": "VI-TAMER",              "color": "darkcyan",   "linestyle": "-"}),
]


def _model_style(family_key: str, fallback_idx: int) -> dict:
    """Return {label, color, linestyle} for a family key."""
    # family_key looks like "ep100/timing-experiment/early_history"
    stem = family_key.replace("_history", "").replace("_episode", "")
    for pattern, style in _MODEL_STYLES:
        if re.search(pattern, stem):
            return style
    # Fallback
    return {
        "label": stem.replace("_", " ").title(),
        "color": _CHART_COLORS[fallback_idx % len(_CHART_COLORS)],
        "linestyle": "-",
    }


def _group_seeds(paths: list[str]) -> list[tuple[str, dict, list[pd.DataFrame]]]:
    """Group CSVs that share the same model family (differ only by seed suffix).

    Returns [(family_key, style_dict, [df, ...])] preserving order of first occurrence.
    """
    import collections
    groups: dict[str, list[pd.DataFrame]] = collections.OrderedDict()
    for p in paths:
        df = _load_history(p)
        if df is None:
            continue
        pp = pathlib.Path(p)
        family = _csv_family(pp)
        groups.setdefault(family, []).append(df)
    result = []
    for i, (fam, dfs) in enumerate(groups.items()):
        style = _model_style(fam, i)
        result.append((fam, style, dfs))
    return result


def generate_chart(chart_type: str, csv_paths: list[str], options: dict) -> str | None:
    """Generate a chart and return base64-encoded PNG string."""
    window = int(options.get("window", 10))

    if chart_type == "training_curves":
        return _chart_training_curves(csv_paths, window)
    elif chart_type == "training_curves_std":
        return _chart_training_curves_std(csv_paths, window)
    elif chart_type == "box_plot":
        return _chart_box_plot(csv_paths)
    elif chart_type == "bar_chart":
        return _chart_bar_chart(csv_paths)
    elif chart_type == "histogram":
        return _chart_histogram(csv_paths)
    elif chart_type == "convergence":
        return _chart_convergence(csv_paths, window)
    elif chart_type == "success_rate":
        return _chart_success_rate(csv_paths, window)
    elif chart_type == "improvement_speed":
        return _chart_improvement_speed(csv_paths, window)
    elif chart_type == "stability":
        return _chart_stability(csv_paths, window)
    elif chart_type == "final_performance":
        return _chart_final_performance(csv_paths, window)
    elif chart_type == "heatmap":
        return _chart_heatmap(csv_paths)
    return None


def _chart_training_curves(paths: list[str], window: int) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        lengths = df["episode_length"].values
        rolled = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
        ax.plot(rolled.index, rolled, label=label, color=color, linewidth=2)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Training Curves (rolling mean, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_training_curves_std(paths: list[str], window: int) -> str:
    grouped = _group_seeds(paths)
    if not grouped:
        return _chart_training_curves(paths, window)

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_c = pd.Series(stacked.mean(axis=0)).rolling(window=window, min_periods=1).mean()
        std_c = pd.Series(stacked.std(axis=0)).rolling(window=window, min_periods=1).mean()
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        color=style["color"], alpha=0.12)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Training Curves (mean ± std, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_box_plot(paths: list[str]) -> str:
    data, labels, colors = [], [], []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        data.append(df["episode_length"].values)
        labels.append(_csv_label(pathlib.Path(p)))
        colors.append(_CHART_COLORS[i % len(_CHART_COLORS)])

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 2), 6))
    bp = ax.boxplot(data, tick_labels=[""] * len(data), patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode Length Distribution — Box Plot")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_bar_chart(paths: list[str]) -> str:
    means, stds, labels, colors = [], [], [], []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        lengths = df["episode_length"].values
        means.append(np.mean(lengths))
        stds.append(np.std(lengths))
        labels.append(_csv_label(pathlib.Path(p)))
        colors.append(_CHART_COLORS[i % len(_CHART_COLORS)])

    fig, ax = plt.subplots(figsize=(max(10, len(means) * 2), 6))
    bars = ax.bar(range(len(means)), means, color=colors, alpha=0.75, yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title("Mean Episode Length ± Std")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _chart_histogram(paths: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    patches_legend = []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        ax.hist(df["episode_length"].values, bins=25, alpha=0.4, color=color, label=label)
        patches_legend.append(mpatches.Patch(facecolor=color, alpha=0.5, label=label))
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.axvline(x=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode Length Distribution — Histogram")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.legend(handles=patches_legend + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_convergence(paths: list[str], window: int) -> str:
    thresholds = [50, 100, 150, 195]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Convergence Analysis — First Episode Crossing Thresholds (rolling window={window})",
        fontsize=13, fontweight="bold",
    )

    crossing_data = {}
    loaded = []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        loaded.append((label, color, df))
        crossings = {}
        rolling = df["episode_length"].rolling(window=window, min_periods=1).mean()
        for th in thresholds:
            crossed = rolling[rolling >= th]
            crossings[th] = int(crossed.index[0]) if not crossed.empty else None
        crossing_data[label] = crossings

    if not loaded:
        plt.close(fig)
        return ""

    # Left: learning curves with threshold markers
    ax = axes[0]
    for label, color, df in loaded:
        rolling = df["episode_length"].rolling(window=window, min_periods=1).mean()
        ax.plot(rolling.index, rolling, color=color, linewidth=2, label=label)
        for th in thresholds:
            ep = crossing_data[label][th]
            if ep is not None:
                ax.plot(ep, th, marker="x", color=color, markersize=8, markeredgewidth=2)
    for th in thresholds:
        c = "red" if th == 195 else "gray"
        ls = "--" if th == 195 else ":"
        ax.axhline(y=th, color=c, linestyle=ls, alpha=0.55, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.98, th + 1, str(th), color=c, fontsize=8, ha="right")
    ax.set_title(f"Learning Curves + Threshold Crossings (× marks)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: grouped bar chart
    ax = axes[1]
    n_models = len(loaded)
    n_th = len(thresholds)
    bar_w = 0.8 / max(n_models, 1)
    x = np.arange(n_th)
    max_ep = 1
    for i, (label, color, df) in enumerate(loaded):
        vals = []
        for th in thresholds:
            ep = crossing_data[label][th]
            val = ep if ep is not None else len(df) + 10
            vals.append(val)
            if ep is not None:
                max_ep = max(max_ep, ep)
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=label, color=color, alpha=0.75)
        for bar, val, th in zip(bars, vals, thresholds):
            ep = crossing_data[label][th]
            txt = str(ep) if ep is not None else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    txt, ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=color, rotation=45)
    ax.set_title("Episode of First Threshold Crossing (lower = faster)")
    ax.set_xlabel("Performance Threshold")
    ax.set_ylabel("Episode Number")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥ {t}" for t in thresholds])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max_ep * 1.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _chart_success_rate(paths: list[str], window: int) -> str:
    """Rolling success rate: % of episodes reaching >=195 steps over a window."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        # For each seed, compute binary success then average across seeds
        success_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            success = (lengths >= 195).astype(float)
            rolled = pd.Series(success).rolling(window=window, min_periods=1).mean() * 100
            success_arrays.append(rolled.values)
        stacked = np.stack(success_arrays)
        mean_c = stacked.mean(axis=0)
        std_c = stacked.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        if n_seeds > 1:
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=style["color"], alpha=0.12)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Success Rate Over Time (rolling {window}-ep window, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_improvement_speed(paths: list[str], window: int) -> str:
    """Episode-over-episode improvement rate (derivative of rolling mean)."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        diff_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            rolled = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
            diff = rolled.diff().fillna(0).values
            diff_arrays.append(diff)
        stacked = np.stack(diff_arrays)
        mean_c = stacked.mean(axis=0)
        std_c = stacked.std(axis=0)
        # Smooth the derivative for readability
        mean_smooth = pd.Series(mean_c).rolling(window=window, min_periods=1).mean()
        std_smooth = pd.Series(std_c).rolling(window=window, min_periods=1).mean()
        x = np.arange(min_len)
        ax.plot(x, mean_smooth, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        if n_seeds > 1:
            ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                            color=style["color"], alpha=0.12)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=1)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Learning Speed — Improvement Rate (Δ rolling mean, window={window}, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Improvement (steps/episode)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_stability(paths: list[str], window: int) -> str:
    """Rolling standard deviation over time — lower = more stable."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        std_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            rolled_std = pd.Series(lengths).rolling(window=window, min_periods=2).std().fillna(0).values
            std_arrays.append(rolled_std)
        stacked = np.stack(std_arrays)
        mean_c = stacked.mean(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Training Stability — Rolling Std Dev (window={window}, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length Std Dev (lower = more stable)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_final_performance(paths: list[str], window: int) -> str:
    """Grouped bar chart: mean of last N episodes for each model family."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    last_n = max(window, 5)
    labels, means, stds, colors = [], [], [], []
    for _fam, style, dfs in grouped:
        # For each seed, take last N episodes, then average across seeds
        seed_means = []
        for df in dfs:
            lengths = df["episode_length"].values
            tail = lengths[-last_n:] if len(lengths) >= last_n else lengths
            seed_means.append(np.mean(tail))
        labels.append(f"{style['label']} (n={len(dfs)})")
        means.append(np.mean(seed_means))
        stds.append(np.std(seed_means) if len(seed_means) > 1 else 0)
        colors.append(style["color"])

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 6))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.75,
                  yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title(f"Final Performance — Mean of Last {last_n} Episodes (across seeds)")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _chart_heatmap(paths: list[str]) -> str:
    """Performance heatmap: model families vs key metrics."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    metric_names = ["Mean", "Median", "Max", "Std", "Success\nRate %", "Best\nWindow"]
    labels = []
    data_rows = []
    colors_list = []
    for _fam, style, dfs in grouped:
        # Combine all episodes across seeds for overall metrics
        all_lengths = np.concatenate([df["episode_length"].values for df in dfs])
        # Best rolling-10 window mean
        if len(all_lengths) >= 10:
            best_win = pd.Series(all_lengths).rolling(10, min_periods=1).mean().max()
        else:
            best_win = np.mean(all_lengths)
        success_rate = (all_lengths >= 195).sum() / len(all_lengths) * 100

        labels.append(style["label"])
        colors_list.append(style["color"])
        data_rows.append([
            np.mean(all_lengths),
            np.median(all_lengths),
            np.max(all_lengths),
            np.std(all_lengths),
            success_rate,
            best_win,
        ])

    data = np.array(data_rows)
    n_models = len(labels)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), max(4, n_models * 0.7 + 2)))

    # Normalize each column for color mapping (0-1)
    data_norm = np.zeros_like(data)
    for j in range(n_metrics):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            data_norm[:, j] = 0.5
    # For Std column (index 3), invert: lower is better
    data_norm[:, 3] = 1.0 - data_norm[:, 3]

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(n_models):
        for j in range(n_metrics):
            val = data[i, j]
            fmt = f"{val:.1f}" if val < 1000 else f"{val:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black" if 0.3 < data_norm[i, j] < 0.7 else "white")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Performance Heatmap (green = better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative (higher = better)")

    plt.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Gameplay chart generation (from live gameplay data, not CSV)
# ---------------------------------------------------------------------------

_GAMEPLAY_CHART_COLORS = [
    "steelblue", "crimson", "green", "darkorange", "purple",
    "teal", "brown", "deeppink", "olive", "slategray",
]


def _gameplay_box_plot(models: list[dict]) -> str:
    n = len(models)
    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    data = [m["history"] for m in models]
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    bp = ax.boxplot(data, tick_labels=[""] * n, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)
    # Goal line removed as requested
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    ax.set_title("Gameplay Performance Distribution")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_bar_chart(models: list[dict]) -> str:
    n = len(models)
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    means = [np.mean(m["history"]) for m in models]
    stds = [np.std(m["history"]) for m in models]
    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    bars = ax.bar(range(n), means, color=colors, alpha=0.75, yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title("Gameplay — Mean Episode Length ± Std")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _gameplay_histogram(models: list[dict]) -> str:
    n = len(models)
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(models):
        ax.hist(m["history"], bins=max(10, len(m["history"]) // 3), alpha=0.4,
                color=colors[i], label=labels[i])
    ax.axvline(x=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.5, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_title("Gameplay — Episode Length Histogram")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_episode_progression(models: list[dict]) -> str:
    """Line chart: episode length over gameplay episodes — shows consistency."""
    n = len(models)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(models):
        color = _GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)]
        h = m["history"]
        ax.plot(range(1, len(h) + 1), h, color=color, linewidth=1.5, alpha=0.7,
                marker="o", markersize=4, label=m["label"])
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title("Gameplay — Episode Progression")
    ax.set_xlabel("Gameplay Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_summary_heatmap(models: list[dict]) -> str:
    """Heatmap comparing models across key gameplay metrics."""
    metric_names = ["Mean", "Median", "Best", "Worst", "Std", "Success\nRate %"]
    labels = [m["label"] for m in models]
    data_rows = []
    for m in models:
        h = np.array(m["history"], dtype=float)
        sr = (h >= 195).sum() / len(h) * 100 if len(h) > 0 else 0
        data_rows.append([np.mean(h), np.median(h), np.max(h), np.min(h), np.std(h), sr])
    data = np.array(data_rows)
    n_models, n_metrics = data.shape

    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.4), max(3.5, n_models * 0.7 + 2)))
    data_norm = np.zeros_like(data)
    for j in range(n_metrics):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            data_norm[:, j] = 0.5
    # Invert Worst (index 3) and Std (index 4): lower is better
    data_norm[:, 3] = 1.0 - data_norm[:, 3]
    data_norm[:, 4] = 1.0 - data_norm[:, 4]

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(n_models):
        for j in range(n_metrics):
            val = data[i, j]
            fmt = f"{val:.1f}" if val < 1000 else f"{val:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black" if 0.3 < data_norm[i, j] < 0.7 else "white")
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Gameplay Performance Heatmap (green = better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative (higher = better)")
    plt.tight_layout()
    return _fig_to_base64(fig)


_GAMEPLAY_CHART_TYPES = {
    "gp_box_plot":       ("Box Plot",              _gameplay_box_plot),
    "gp_bar_chart":      ("Bar Chart (Mean ± Std)", _gameplay_bar_chart),
    "gp_histogram":      ("Histogram",              _gameplay_histogram),
    "gp_progression":    ("Episode Progression",    _gameplay_episode_progression),
    "gp_heatmap":        ("Performance Heatmap",    _gameplay_summary_heatmap),
}


@app.route("/api/gameplay-chart", methods=["POST"])
def api_gameplay_chart():
    """Generates comparison charts from live gameplay/simulation data."""
    data = request.get_json(force=True)
    # Support both 'summary' (from live sim) and 'models' (from legacy/free play)
    models = data.get("summary") or data.get("models")
    chart_types = data.get("chart_types")
    
    if not models:
        return jsonify({"error": "No data"}), 400

    # If simple request (just one image), return it directly for compatibility
    if not chart_types:
        return jsonify({"image": _gameplay_box_plot(models)})

    # Else return multiple charts
    results = []
    for ct in chart_types:
        if ct not in _GAMEPLAY_CHART_TYPES: continue
        nice_name, fn = _GAMEPLAY_CHART_TYPES[ct]
        try:
            img = fn(models)
            results.append({"chart_type": ct, "title": nice_name, "image": img})
        except Exception as exc:
            results.append({"chart_type": ct, "title": nice_name, "error": str(exc)})
    return jsonify({"charts": results})


# ---------------------------------------------------------------------------
# Chart API routes
# ---------------------------------------------------------------------------

@app.route("/api/csvs")
def api_csvs():
    return jsonify(scan_csvs())


@app.route("/api/chart", methods=["POST"])
def api_chart():
    data = request.get_json(force=True)
    chart_type = data.get("chart_type", "training_curves")
    csv_paths = data.get("csvs", [])
    options = data.get("options", {})

    if not csv_paths:
        return jsonify({"error": "No CSVs selected"}), 400

    try:
        img_b64 = generate_chart(chart_type, csv_paths, options)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not img_b64:
        return jsonify({"error": "Chart generation failed — no data"}), 400

    return jsonify({"image": img_b64, "chart_type": chart_type})


_ALL_CHART_TYPES = [
    "training_curves",
    "training_curves_std",
    "box_plot",
    "bar_chart",
    "histogram",
    "convergence",
    "success_rate",
    "improvement_speed",
    "stability",
    "final_performance",
    "heatmap",
]


@app.route("/api/multi-chart", methods=["POST"])
def api_multi_chart():
    """Generate several charts in one request and return a list of results."""
    data = request.get_json(force=True)
    csv_paths = data.get("csvs", [])
    chart_types = data.get("chart_types", _ALL_CHART_TYPES)
    options = data.get("options", {})

    if not csv_paths:
        return jsonify({"error": "No CSVs selected"}), 400

    results = []
    for ct in chart_types:
        try:
            img_b64 = generate_chart(ct, csv_paths, options)
            if img_b64:
                results.append({"chart_type": ct, "image": img_b64})
            else:
                results.append({"chart_type": ct, "error": "No data"})
        except Exception as exc:
            results.append({"chart_type": ct, "error": str(exc)})

    return jsonify({"charts": results})


# ---------------------------------------------------------------------------
# Single-file HTML / CSS / JS frontend
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CartPole HCRL — Interactive Insights</title>
<style>
:root {
  --bg: #0f172a;
  --card-bg: rgba(30, 41, 59, 0.7);
  --sidebar-bg: rgba(15, 23, 42, 0.5);
  --accent: #38bdf8;
  --accent-glow: rgba(56, 189, 248, 0.3);
  --danger: #ef4444;
  --success: #22c55e;
  --text: #f8fafc;
  --text-muted: #94a3b8;
  --border: rgba(255, 255, 255, 0.1);
  --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  margin: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background-color: var(--bg); color: var(--text);
  background-image: radial-gradient(circle at 50% -20%, #1e293b 0%, #0f172a 100%);
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}

header {
  padding: 1rem 5%; display: flex; align-items: center; gap: 2rem;
  border-bottom: 1px solid var(--border); backdrop-filter: blur(10px); flex-shrink: 0;
}
.hdr-logo { height: 45px; filter: drop-shadow(0 0 10px var(--accent-glow)); }
.hdr-text h1 { margin: 0; font-size: 1.5rem; background: linear-gradient(90deg, #fff, var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hdr-text p { margin: 2px 0 0; color: var(--text-muted); font-size: 0.8rem; }

.tab-bar {
  display: flex; gap: 0.5rem; padding: 0.5rem 5%; background: rgba(15, 23, 42, 0.5);
  backdrop-filter: blur(15px); border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.tab-btn {
  background: none; border: none; color: var(--text-muted); padding: 0.5rem 1rem;
  cursor: pointer; font-size: 0.9rem; font-weight: 500; transition: all 0.2s;
  border-radius: 8px; display: flex; align-items: center; gap: 8px;
}
.tab-btn:hover { background: rgba(255,255,255,0.05); color: #fff; }
.tab-btn.active { background: var(--accent); color: #000; font-weight: 600; box-shadow: 0 0 15px var(--accent-glow); }

.container { flex: 1; overflow-y: auto; padding: 2rem 5%; position: relative; }
.tab-page { display: none; animation: fadeIn 0.4s ease-out; }
.tab-page.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* Experiment Grid */
.exp-grid { display: flex; flex-direction: column; gap: 2.5rem; max-width: 1200px; margin: 0 auto; }
.exp-card {
  background: var(--card-bg); border-radius: 20px; border: 1px solid var(--border);
  overflow: hidden; backdrop-filter: blur(10px); box-shadow: var(--shadow);
}
.exp-header { padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); background: rgba(15, 23, 42, 0.3); }
.exp-header h2 { margin: 0; font-size: 1.4rem; color: var(--accent); }
.exp-header .subtitle { color: var(--text-muted); font-size: 0.8rem; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.exp-body { display: flex; flex-direction: column; gap: 2rem; padding: 2rem; }
.exp-info { width: 100%; }
.exp-info p { color: #cbd5e1; line-height: 1.6; font-size: 0.95rem; margin-bottom: 2rem; max-width: 800px; }

.btn {
  padding: 0.7rem 1.4rem; border-radius: 10px; border: none; font-weight: 600;
  cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 8px;
  font-size: 0.9rem;
}
.btn-primary { background: var(--accent); color: #000; }
.btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 5px 15px var(--accent-glow); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

/* Visualizer Cards */
.vis-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
.mini-card {
  background: rgba(15, 23, 42, 0.5); border-radius: 12px; border: 1px solid var(--border); padding: 12px;
  text-align: center; display: flex; flex-direction: column; transition: transform 0.2s;
}
.mini-card:hover { transform: translateY(-5px); border-color: var(--accent); }
.mini-card .canvas-wrap { aspect-ratio: 4/3; background: #000; border-radius: 8px; margin-bottom: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.mini-card img { width: 100%; height: 100%; object-fit: cover; }
.mini-card .label { font-size: 0.75rem; font-weight: 700; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mini-card .val { font-size: 1.1rem; font-weight: 800; color: var(--accent); border-top: 1px solid var(--border); margin-top: 6px; padding-top: 6px; }

.vis-container { display: flex; flex-direction: column; align-items: center; gap: 1.5rem; width: 100%; }

/* Charts Area */
.chart-area { min-height: 450px; width: 100%; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); border-radius: 15px; border: 1px solid var(--border); overflow: hidden; }
.chart-area img { width: 100%; height: auto; max-height: 700px; object-fit: contain; }
.chart-area img:hover { transform: scale(1.02); }

/* Play Tab Specific */
.play-layout { display: grid; grid-template-columns: 300px 1fr; gap: 2rem; height: calc(100vh - 180px); }
.sidebar { background: var(--card-bg); border-radius: 20px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
.sb-head { padding: 1.2rem; border-bottom: 1px solid var(--border); font-weight: 700; display: flex; justify-content: space-between; }
.model-list { flex: 1; overflow-y: auto; padding: 0.5rem; }
.grp-hdr { padding: 10px 12px 5px; font-size: 0.7rem; font-weight: 800; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; }
.model-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
.model-item:hover { background: rgba(255,255,255,0.05); }
.model-item input { accent-color: var(--accent); }
.model-item span { font-size: 0.85rem; }
.sb-ctrls { padding: 1.2rem; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 1rem; }

.main-game { background: rgba(0,0,0,0.2); border-radius: 20px; border: 1px solid var(--border); position: relative; overflow-y: auto; padding: 1.5rem; }

/* Results Table */
.res-table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-size: 0.85rem; }
.res-table th { text-align: left; padding: 10px; border-bottom: 2px solid var(--border); color: var(--text-muted); }
.res-table td { padding: 10px; border-bottom: 1px solid var(--border); }
.badge-best { background: var(--success); color: #000; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 800; margin-left: 6px; }

/* Global Utilities */
.hint { text-align: center; color: var(--text-muted); padding: 4rem 2rem; border: 2px dashed rgba(255,255,255,0.05); border-radius: 24px; margin: 2rem; }
.spinner { width: 30px; height: 30px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Game Grid for Play tab */
.game-grid { display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.game-card { background: var(--card-bg); border-radius: 15px; border: 1px solid var(--border); padding: 1rem; display: flex; flex-direction: column; gap: 10px; }
.game-card .frame-box { aspect-ratio: 4/3; background: #000; border-radius: 10px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
.game-card img { width: 100%; height: 100%; object-fit: cover; }
.game-card .card-head { display: flex; justify-content: space-between; align-items: center; }
.game-card .stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; border-top: 1px solid var(--border); padding-top: 10px; }
.game-card .stat-box { text-align: center; }
.game-card .stat-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
.game-card .stat-lbl { font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; }

/* Statusbar */
.statusbar { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px 5%; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(10px); border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-muted); z-index: 1000; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #475569; display: inline-block; margin-right: 8px; }
.dot.live { background: var(--success); box-shadow: 0 0 10px var(--success); animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
</style>
</head>
<body>

<header>
  <img src="/logo" class="hdr-logo" alt="HUST" onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/a/a2/Logo_Hust.png'">
  <div class="hdr-text">
    <h1>CartPole HCRL — Interactive Insights</h1>
    <p>Hanoi University of Science and Technology &nbsp;·&nbsp; Statistical Machine Learning Dashboard</p>
  </div>
</header>

<nav class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('report')" id="tabReport">
    📖 <span>Experiments</span>
  </button>
  <button class="tab-btn" onclick="switchTab('play')" id="tabPlay">
    🎮 <span>Free Play</span>
  </button>
  <button class="tab-btn" onclick="switchTab('charts')" id="tabCharts">
    📊 <span>Raw Charts</span>
  </button>
</nav>

<div class="container">
  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- TAB 0: EXPERIMENT REPORT                                      -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <div class="tab-page active" id="pageReport">
    <div class="exp-grid" id="expGrid">
      <div class="hint"><div class="spinner"></div><p>Loading experiments...</p></div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- TAB 1: FREE PLAY                                              -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <div class="tab-page" id="pagePlay">
    <div class="play-layout">
      <aside class="sidebar">
        <div class="sb-head">Models <span id="selCount" style="color:var(--accent)">0</span></div>
        <div class="model-list" id="modelList"></div>
        <div class="sb-ctrls">
          <div style="font-size: 0.8rem;">
            <div style="display:flex; justify-content:space-between"><span>Episodes</span> <strong id="epVal">5</strong></div>
            <input type="range" id="epSlider" min="1" max="50" value="5" style="width:100%" oninput="document.getElementById('epVal').textContent=this.value">
          </div>
          <button class="btn btn-primary" id="playBtn" onclick="togglePlay()" style="justify-content:center">▶ Start Watch</button>
        </div>
      </aside>
      <main class="main-game">
        <div id="gameArea">
          <div class="hint">Select models from the sidebar to watch them perform live.</div>
        </div>
        <div id="resultsTableContainer"></div>
      </main>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- TAB 2: RAW CHARTS                                             -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <div class="tab-page" id="pageCharts">
    <div class="play-layout">
      <aside class="sidebar">
        <div class="sb-head">Data Sources <span id="csvCount" style="color:var(--accent)">0</span></div>
        <div class="model-list" id="csvList">
          <div class="hint" style="padding:1rem; border:none">Loading data...</div>
        </div>
        <div class="sb-ctrls">
          <div class="ctrl-group">
            <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:8px">Select Chart Types</div>
            <div id="typeList" style="display:flex; flex-direction:column; gap:4px">
              <label class="model-item"><input type="checkbox" value="training_curves" checked> <span>Learning Curves</span></label>
              <label class="model-item"><input type="checkbox" value="training_curves_std"> <span>Curves (Mean±Std)</span></label>
              <label class="model-item"><input type="checkbox" value="box_plot"> <span>Distribution (Box)</span></label>
              <label class="model-item"><input type="checkbox" value="convergence"> <span>Convergence Analysis</span></label>
              <label class="model-item"><input type="checkbox" value="success_rate"> <span>Success Rate</span></label>
              <label class="model-item"><input type="checkbox" value="heatmap"> <span>Performance Heatmap</span></label>
            </div>
          </div>
          <button class="btn btn-primary" id="genBtn" onclick="generateCharts()" style="justify-content:center">📊 Generate Charts</button>
        </div>
      </aside>
      <main class="main-game" id="chartDisplay">
        <div class="hint">
          <div class="hint-icon" style="font-size:3rem; margin-bottom:1rem">📈</div>
          <p>Select training history files (CSV) and chart types to perform deep analysis.</p>
        </div>
      </main>
    </div>
  </div>
</div>

<div class="statusbar">
  <div><span class="dot" id="statusDot"></span><span id="statusTxt">System Ready</span></div>
  <div id="progressTxt">Waiting for user input...</div>
</div>

<script>
/* ══════════════════════════════════════════════════════════════════
   GLOBAL STATE & UTILS
   ══════════════════════════════════════════════════════════════════ */
let experiments = [];
let modelsLoaded = false;
let selectedModels = new Set();
let es = null;

function esc(s) {
  if (!s) return "";
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  document.getElementById('tab' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
  document.getElementById('page' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
  
  if (tab === 'report') loadExperiments();
  if (tab === 'play' && !modelsLoaded) loadModels();
  if (tab === 'charts' && !chartsLoaded) loadCsvs();
}

/* ══════════════════════════════════════════════════════════════════
   RAW CHARTS TAB
   ══════════════════════════════════════════════════════════════════ */
let chartsLoaded = false;
let selectedCsvs = new Set();

function loadCsvs() {
  chartsLoaded = true;
  fetch('/api/csvs')
    .then(r => r.json())
    .then(csvs => {
      const list = document.getElementById('csvList');
      if (!csvs.length) { list.innerHTML = '<div class="hint">No CSV files found.</div>'; return; }
      
      const tree = {};
      csvs.forEach(c => { if (!tree[c.group]) tree[c.group] = []; tree[c.group].push(c); });
      
      let html = '';
      for (const [grp, items] of Object.entries(tree)) {
        html += `<div class="grp-hdr">${esc(grp)}</div>`;
        items.forEach(c => {
          html += `
            <label class="model-item">
              <input type="checkbox" value="${esc(c.path)}" onchange="onCsvToggle(this)">
              <span>${esc(c.label)}</span>
            </label>`;
        });
      }
      list.innerHTML = html;
    });
}

function onCsvToggle(cb) {
  cb.checked ? selectedCsvs.add(cb.value) : selectedCsvs.delete(cb.value);
  document.getElementById('csvCount').textContent = selectedCsvs.size;
}

function generateCharts() {
  const types = [...document.querySelectorAll('#typeList input:checked')].map(i => i.value);
  if (!selectedCsvs.size) { alert("Select at least one CSV file."); return; }
  if (!types.length) { alert("Select at least one chart type."); return; }

  const btn = document.getElementById('genBtn');
  const display = document.getElementById('chartDisplay');
  
  btn.disabled = true;
  btn.textContent = "⌛ Generating...";
  display.innerHTML = '<div class="hint"><div class="spinner"></div><p>Performing analysis...</p></div>';

  fetch('/api/multi-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      csvs: [...selectedCsvs],
      chart_types: types,
      options: { window: 20 }
    })
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    btn.textContent = "📊 Generate Charts";
    
    if (!data.charts || !data.charts.length) {
      display.innerHTML = '<div class="hint">No data could be generated.</div>';
      return;
    }

    display.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:20px; padding:10px">
        ${data.charts.map(c => `
          <div class="exp-card">
            <div class="exp-header"><h2>${esc(c.chart_type.replace(/_/g, ' ').toUpperCase())}</h2></div>
            <div class="chart-area">${c.image ? `<img src="data:image/png;base64,${c.image}">` : `<div style="color:var(--danger)">Error: ${esc(c.error)}</div>`}</div>
          </div>
        `).join('')}
      </div>`;
  })
  .catch(err => {
    btn.disabled = false;
    btn.textContent = "📊 Generate Charts";
    alert("Request failed: " + err.message);
  });
}

/* ══════════════════════════════════════════════════════════════════
   LANDING PAGE: EXPERIMENTS
   ══════════════════════════════════════════════════════════════════ */
function loadExperiments() {
  fetch('/api/experiments')
    .then(r => r.json())
    .then(data => {
      experiments = data;
      const grid = document.getElementById('expGrid');
      grid.innerHTML = data.map(exp => `
        <div class="exp-card">
          <div class="exp-header">
            <div class="subtitle">${esc(exp.subtitle)}</div>
            <h2>${esc(exp.title)}</h2>
          </div>
          <div class="exp-body">
            <div class="exp-info">
              <p>${esc(exp.description)}</p>
              <div class="vis-container">
                <div class="vis-grid">
                  ${exp.actual_models.map((m, i) => `
                    <div class="mini-card">
                      <div class="canvas-wrap"><img id="f-${exp.id}-${i}" src="" style="display:none"></div>
                      <div class="label" title="${esc(m.label)}">${esc(m.label)}</div>
                      <div class="val" id="v-${exp.id}-${i}">0 steps</div>
                    </div>
                  `).join('')}
                </div>
                <button class="btn btn-primary" onclick="runExp('${exp.id}', this)">▶ Run Experiment Simulation</button>
              </div>
            </div>
            <div class="exp-charts">
              <div class="chart-area" id="c-${exp.id}">
                <div style="color:var(--text-muted); font-size:0.9rem; text-align:center; padding: 2rem;">
                  <p>Click "Run" to view live evaluation results here.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      `).join('');
    });
}

function runExp(id, btn) {
  const exp = experiments.find(e => e.id === id);
  if (!exp) return;

  btn.disabled = true;
  btn.textContent = "⌛ Simulating...";
  
  const p = new URLSearchParams();
  exp.actual_models.forEach(m => p.append('models', m.path));
  p.set('episodes', 10);
  p.set('fps', 35);

  const eventSource = new EventSource('/api/play?' + p);
  eventSource.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'frame') {
      d.frames.forEach((b64, i) => {
        const img = document.getElementById(`f-${id}-${i}`);
        if (img) { img.src = 'data:image/jpeg;base64,' + b64; img.style.display = 'block'; }
        const val = document.getElementById(`v-${id}-${i}`);
        if (val && d.stats) val.textContent = d.stats[i].steps + ' steps';
      });
    } else if (d.type === 'done') {
      eventSource.close();
      btn.textContent = "✅ Simulation Done";
      fetchLiveChart(exp.id, d.summary);
    }
  };
  eventSource.onerror = () => { eventSource.close(); btn.disabled = false; btn.textContent = "▶ Run Experiment"; };
}

function fetchLiveChart(id, summary) {
  fetch('/api/gameplay-chart', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ summary: summary })
  })
  .then(r => r.json())
  .then(data => {
    if (data.image) {
      const area = document.getElementById('c-' + id);
      const liveDiv = document.createElement('div');
      liveDiv.innerHTML = `
        <div style="margin-bottom: 2rem; border: 2px solid var(--accent); border-radius: 15px; overflow: hidden; background: rgba(56, 189, 248, 0.05)">
          <div style="padding: 10px 20px; background: var(--accent); color: #000; font-weight: 800; font-size: 0.8rem">LIVE PERFORMANCE RESULTS (Current Evaluation)</div>
          <img src="data:image/png;base64,${data.image}" style="width:100%; height:auto; display:block">
        </div>
      `;
      area.prepend(liveDiv);
    }
  });
}

function fetchExpCharts(exp) {
  const area = document.getElementById('c-' + exp.id);
  area.innerHTML = '<div class="spinner"></div>';
  
  fetch('/api/multi-chart', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      csvs: exp.actual_csvs,
      chart_types: exp.chart_types,
      options: { window: 15 }
    })
  })
  .then(r => r.json())
  .then(data => {
    if (data.charts && data.charts.length > 0) {
      area.innerHTML = `<img src="data:image/png;base64,${data.charts[0].image}">`;
    } else {
      area.innerHTML = '<div style="color:var(--danger)">No chart data found.</div>';
    }
  });
}

/* ══════════════════════════════════════════════════════════════════
   FREE PLAY TAB
   ══════════════════════════════════════════════════════════════════ */
function loadModels() {
  modelsLoaded = true;
  fetch('/api/models')
    .then(r => r.json())
    .then(models => {
      const list = document.getElementById('modelList');
      if (!models.length) { list.innerHTML = '<div class="hint">No models found.</div>'; return; }
      
      const tree = {};
      models.forEach(m => { if (!tree[m.group]) tree[m.group] = []; tree[m.group].push(m); });
      
      let html = '';
      for (const [grp, items] of Object.entries(tree)) {
        html += `<div class="grp-hdr">${esc(grp)}</div>`;
        items.forEach(m => {
          html += `
            <label class="model-item">
              <input type="checkbox" value="${esc(m.path)}" data-label="${esc(m.label)}" onchange="onModelToggle(this)">
              <span>${esc(m.label)}</span>
            </label>`;
        });
      }
      list.innerHTML = html;
    });
}

function onModelToggle(cb) {
  cb.checked ? selectedModels.add(cb.value) : selectedModels.delete(cb.value);
  document.getElementById('selCount').textContent = selectedModels.size;
}

function togglePlay() {
  if (es) { stopPlay(); } else { startPlay(); }
}

function startPlay() {
  if (!selectedModels.size) { alert("Please select at least one model."); return; }
  
  const paths = [...selectedModels];
  const episodes = document.getElementById('epSlider').value;
  const p = new URLSearchParams();
  paths.forEach(path => p.append('models', path));
  p.set('episodes', episodes);
  p.set('fps', 30);
  
  buildGameGrid(paths);
  
  const btn = document.getElementById('playBtn');
  btn.textContent = "◼ Stop Simulation";
  btn.classList.add('btn-stop');
  document.getElementById('statusDot').classList.add('live');
  document.getElementById('statusTxt').textContent = "Streaming Live Gameplay...";

  es = new EventSource('/api/play?' + p);
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'frame') {
      d.frames.forEach((b64, i) => {
        const img = document.getElementById(`live-f-${i}`);
        if (img) { img.src = 'data:image/jpeg;base64,' + b64; img.style.display = 'block'; }
        const steps = document.getElementById(`live-s-${i}`);
        if (steps) steps.textContent = d.stats[i].steps;
        const mean = document.getElementById(`live-m-${i}`);
        if (mean) mean.textContent = d.stats[i].mean.toFixed(1);
      });
      document.getElementById('progressTxt').textContent = `Episode ${d.episode} / ${d.total}`;
    } else if (d.type === 'done') {
      stopPlay();
      renderFinalResults(d.summary);
    }
  };
  es.onerror = () => stopPlay();
}

function stopPlay() {
  if (es) { es.close(); es = null; }
  const btn = document.getElementById('playBtn');
  btn.textContent = "▶ Start Watch";
  btn.classList.remove('btn-stop');
  document.getElementById('statusDot').classList.remove('live');
  document.getElementById('statusTxt').textContent = "System Ready";
}

function buildGameGrid(paths) {
  const container = document.getElementById('gameArea');
  container.innerHTML = `
    <div class="game-grid">
      ${paths.map((p, i) => `
        <div class="game-card">
          <div class="card-head"><strong>Model ${i+1}</strong></div>
          <div class="frame-box"><img id="live-f-${i}" src="" style="display:none"></div>
          <div class="stats-row">
            <div class="stat-box"><div class="stat-val" id="live-s-${i}">0</div><div class="stat-lbl">Steps</div></div>
            <div class="stat-box"><div class="stat-val" id="live-m-${i}">0</div><div class="stat-lbl">Mean</div></div>
            <div class="stat-box"><div class="stat-val">—</div><div class="stat-lbl">Goal</div></div>
          </div>
        </div>
      `).join('')}
    </div>`;
}

function renderFinalResults(summary) {
  const container = document.getElementById('resultsTableContainer');
  const best = Math.max(...summary.map(s => s.mean));
  container.innerHTML = `
    <table class="res-table">
      <thead><tr><th>Model</th><th>Mean</th><th>Best</th><th>Worst</th><th>Success Rate</th></tr></thead>
      <tbody>
        ${summary.map(s => `
          <tr>
            <td><strong>${esc(s.label)}</strong>${s.mean === best ? '<span class="badge-best">TOP</span>' : ''}</td>
            <td style="color:var(--accent); font-weight:700">${s.mean}</td>
            <td>${s.best}</td>
            <td>${s.worst}</td>
            <td>${s.goal_rate}%</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

// Initial Load
loadExperiments();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # Allow ngrok / any reverse-proxy host header
    app.config["SERVER_NAME"] = None
    print("=" * 50)
    print("  CartPole HCRL Visualizer")
    print("  Local:  http://localhost:5000")
    print("  Expose: ngrok http 5000")
    print("=" * 50)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
