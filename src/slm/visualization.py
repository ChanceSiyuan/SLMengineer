"""Plotting utilities for SLM hologram analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_phase(
    phase: np.ndarray,
    title: str = "Phase",
    ax: plt.Axes | None = None,
    colorbar: bool = True,
) -> plt.Axes:
    """Plot a phase map with cyclic colormap."""
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, label="Phase (rad)")
    return ax


def plot_intensity(
    field: np.ndarray,
    title: str = "Intensity",
    ax: plt.Axes | None = None,
    log_scale: bool = False,
    colorbar: bool = True,
) -> plt.Axes:
    """Plot |E|^2 intensity map."""
    if ax is None:
        _, ax = plt.subplots()
    intensity = np.abs(field) ** 2
    if log_scale:
        intensity = np.log10(intensity + 1e-30)
    im = ax.imshow(intensity, cmap="hot")
    ax.set_title(title)
    if colorbar:
        label = "log10(I)" if log_scale else "Intensity"
        plt.colorbar(im, ax=ax, label=label)
    return ax


def plot_convergence(
    history: list[float],
    ylabel: str = "Non-uniformity",
    title: str = "Convergence",
    ax: plt.Axes | None = None,
    semilogy: bool = True,
) -> plt.Axes:
    """Plot a convergence curve (metric vs iteration)."""
    if ax is None:
        _, ax = plt.subplots()
    if semilogy and all(v > 0 for v in history):
        ax.semilogy(history)
    else:
        ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_comparison(
    results: dict[str, list[float]],
    ylabel: str = "Non-uniformity",
    title: str = "Algorithm Comparison",
) -> plt.Figure:
    """Compare convergence of multiple algorithms on the same axes."""
    fig, ax = plt.subplots()
    for label, history in results.items():
        if all(v > 0 for v in history):
            ax.semilogy(history, label=label)
        else:
            ax.plot(history, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_spot_histogram(
    intensities: np.ndarray,
    title: str = "Spot Intensity Histogram",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of spot intensities with uniformity annotation."""
    if ax is None:
        _, ax = plt.subplots()
    mean_val = np.mean(intensities)
    std_val = np.std(intensities)
    ax.hist(intensities, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.4f}")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    nu = std_val / mean_val if mean_val > 0 else 0
    ax.set_title(f"{title}\nNon-uniformity: {nu:.2%}")
    ax.legend()
    return ax


def plot_hologram_summary(
    slm_phase: np.ndarray,
    focal_field: np.ndarray,
    target: np.ndarray,
) -> plt.Figure:
    """Four-panel summary: SLM phase, focal intensity, target, difference."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plot_phase(slm_phase, "SLM Phase", ax=axes[0, 0])
    plot_intensity(focal_field, "Focal Intensity", ax=axes[0, 1])
    plot_intensity(target, "Target", ax=axes[1, 0])

    # Difference
    diff = np.abs(focal_field) ** 2 - np.abs(target) ** 2
    im = axes[1, 1].imshow(diff, cmap="RdBu_r")
    axes[1, 1].set_title("Intensity Difference")
    plt.colorbar(im, ax=axes[1, 1])

    fig.tight_layout()
    return fig
