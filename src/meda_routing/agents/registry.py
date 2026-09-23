"""Registry of policy feature extractors.

The paper's agent is a CNN over the image observation.  New architectures —
e.g. a graph neural network over the microelectrode grid — plug in by
registering an SB3 ``BaseFeaturesExtractor`` subclass under a name and
selecting it with ``agent.extractor`` in a training config::

    from meda_routing.agents import register_extractor

    @register_extractor("gnn")
    class MedaGNN(BaseFeaturesExtractor):
        ...

See ``docs/EXTENDING.md``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .cnn import MedaCNN

_EXTRACTORS: Dict[str, Type[BaseFeaturesExtractor]] = {"cnn": MedaCNN}


def register_extractor(name: str) -> Callable[[Type[BaseFeaturesExtractor]], Type[BaseFeaturesExtractor]]:
    def decorator(cls: Type[BaseFeaturesExtractor]) -> Type[BaseFeaturesExtractor]:
        if name in _EXTRACTORS and _EXTRACTORS[name] is not cls:
            raise ValueError(f"feature extractor {name!r} is already registered")
        _EXTRACTORS[name] = cls
        return cls

    return decorator


def get_extractor(name: str) -> Type[BaseFeaturesExtractor]:
    try:
        return _EXTRACTORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown feature extractor {name!r}; known: {sorted(_EXTRACTORS)}") from exc


def available_extractors() -> Dict[str, Type[BaseFeaturesExtractor]]:
    return dict(_EXTRACTORS)


def policy_kwargs_for(name: str, extractor_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """SB3 ``policy_kwargs`` with a shared extractor and linear actor/critic heads."""
    return {
        "features_extractor_class": get_extractor(name),
        "features_extractor_kwargs": dict(extractor_kwargs or {}),
        "net_arch": dict(pi=[], vf=[]),
        "share_features_extractor": True,
    }
