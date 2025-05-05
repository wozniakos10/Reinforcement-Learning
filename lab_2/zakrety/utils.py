from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
import os
import problem


def draw_arrow(axes: plt.Axes, begin: tuple[int, int], end: tuple[int, int]) -> None:
    (begin_y, begin_x), (end_y, end_x) = begin, end
    delta_x, delta_y = end_x - begin_x, end_y - begin_y
    axes.arrow(
        begin_x + 0.5,
        begin_y + 0.5,
        delta_x,
        delta_y,
        length_includes_head=True,
        head_width=0.8,
        head_length=0.8,
        fc="r",
        ec="r",
    )


def draw_episode(
    track: np.ndarray, positions: list[problem.Position], episode: int, corner_name: str, is_validation: bool = False
) -> None:
    ax = plt.axes()
    ax.imshow(track)
    for i in range(len(positions) - 1):
        begin, end = positions[i], positions[i + 1]
        draw_arrow(ax, begin, end)
    if not os.path.exists(f"plots/{corner_name}"):
        os.makedirs(f"plots/{corner_name}")
    if not os.path.exists(f"plots/{corner_name}/validation"):
        os.makedirs(f"plots/{corner_name}/validation")
    if is_validation:
        plt.savefig(f"plots/{corner_name}/validation/validation_{episode}.png", dpi=300)
    else:
        plt.savefig(f"plots/{corner_name}/track_{episode}.png", dpi=300)
    plt.clf()


def draw_penalties_plot(penalties: list[int], window_size: int, episode: int, corner_name: str) -> None:
    means = [np.mean(penalties[i : i + window_size]) for i in range(len(penalties) - window_size)]
    ax = plt.axes()
    ax.plot(means)
    ax.set_xlabel("Liczba epizodów")
    ax.set_ylabel("Funkcja kary")
    plt.title("Wartość funkcji kary")
    plt.savefig(f"plots/{corner_name}/penalties_{episode}.png", dpi=300)
    plt.clf()
