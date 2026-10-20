# Plotting helpers shared by EDA-related scripts.

from typing import Any

from matplotlib.colors import BoundaryNorm, ListedColormap


def build_discrete_plot_kwargs(colors: list[str]) -> dict[str, Any]:
    """
    imshow kwargs for images whose pixel values are discrete categories 0..N-1.
    Boundaries [0, 1, ..., N] put each integer value in its own color bin.
    """
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(colors) + 1)), cmap.N)
    return {"cmap": cmap, "norm": norm}
