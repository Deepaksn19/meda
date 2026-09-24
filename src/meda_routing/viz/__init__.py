"""Figures and animations in the style of the paper (Figs. 4, 7, 8, 9 and 14).

* :mod:`.plots`: training curves (:func:`plot_training_curves`,
  :func:`plot_training_comparison`, :func:`plot_training_panels`), bioassay
  completion probability versus cycles (:func:`plot_completion_cdf`) and
  droplet trajectories on the health map (:func:`plot_routing_path`), plus
  the ``draw_*`` building blocks that draw into an existing axes.
* :mod:`.animation`: GIFs of routing episodes (:func:`record_episode`).

Everything renders through matplotlib's Agg / PDF canvases without pyplot,
so it runs headless.
"""

from .animation import frame_scale, record_episode
from .plots import (
    PAPER_ROUTER_STYLES,
    TRAINING_LABELS,
    aggregate_histories,
    draw_completion_cdf,
    draw_routing_path,
    draw_training,
    load_history,
    plot_completion_cdf,
    plot_routing_path,
    plot_training_comparison,
    plot_router_comparison,
    plot_training_curves,
    plot_training_panels,
    router_styles,
)

__all__ = [
    "PAPER_ROUTER_STYLES",
    "TRAINING_LABELS",
    "aggregate_histories",
    "draw_completion_cdf",
    "draw_routing_path",
    "draw_training",
    "frame_scale",
    "load_history",
    "plot_completion_cdf",
    "plot_routing_path",
    "plot_training_comparison",
    "plot_router_comparison",
    "plot_training_curves",
    "plot_training_panels",
    "record_episode",
    "router_styles",
]
