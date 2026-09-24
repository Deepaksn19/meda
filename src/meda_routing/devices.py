"""Choosing the compute device: a local GPU when there is one, else the CPU.

Every command that builds or loads a network (``meda train``, ``evaluate``,
``compare``, ``bioassay``, ``render``) passes its device setting through
:func:`resolve_device`:

* ``"auto"`` (the default): the local CUDA GPU with the most free memory,
  else Apple's MPS backend, else the CPU;
* ``"cuda"`` / ``"cuda:1"`` / ``"mps"`` / ``"cpu"``: that device.  A GPU that
  is not available falls back to the CPU with a warning instead of crashing;
* the environment variable ``MEDA_DEVICE`` overrides whatever the config or
  command line says, e.g. ``MEDA_DEVICE=cuda:1 meda train ...``.

To train on the GPU of *another* machine (e.g. a lab server at 192.168.x.x),
run the command there with ``scripts/run_on_gpu_server.sh``; on that server
this module then picks its GPU.  ``meda devices`` lists what is available.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List

import torch

#: Environment variable that overrides the configured device.
DEVICE_ENV_VAR = "MEDA_DEVICE"


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def cuda_devices() -> List[Dict[str, object]]:
    """Local CUDA GPUs with their name and total / free memory (GiB)."""
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except (RuntimeError, AttributeError):  # busy or very old torch
            free, total = props.total_memory, props.total_memory
        out.append({
            "device": f"cuda:{i}",
            "name": props.name,
            "total_gib": total / 2**30,
            "free_gib": free / 2**30,
        })
    return out


def available_devices() -> List[Dict[str, object]]:
    """Every usable device, best first: CUDA GPUs, MPS, CPU."""
    devices = sorted(cuda_devices(), key=lambda d: -float(d["free_gib"]))
    if _mps_available():
        devices.append({"device": "mps", "name": "Apple GPU (MPS)"})
    devices.append({"device": "cpu", "name": f"CPU ({os.cpu_count()} logical cores)"})
    return devices


def resolve_device(device: str = "auto") -> str:
    """The torch device string to use for ``device`` (see the module docstring)."""
    requested = os.environ.get(DEVICE_ENV_VAR) or device or "auto"
    requested = str(requested).strip().lower()
    if requested == "auto":
        gpus = cuda_devices()
        if gpus:
            return str(max(gpus, key=lambda d: float(d["free_gib"]))["device"])
        return "mps" if _mps_available() else "cpu"
    if requested.startswith("cuda"):
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return f"cuda:{index}"
        warnings.warn(f"{requested} requested but not available here; using the CPU", stacklevel=2)
        return "cpu"
    if requested == "mps":
        if _mps_available():
            return "mps"
        warnings.warn("mps requested but not available here; using the CPU", stacklevel=2)
        return "cpu"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"unknown device {device!r}: use auto, cpu, cuda, cuda:<index> or mps")


def describe_devices() -> str:
    """Human-readable list of devices and the one ``auto`` picks."""
    lines = []
    for d in available_devices():
        mem = f"  {d['free_gib']:.1f} / {d['total_gib']:.1f} GiB free" if "free_gib" in d else ""
        lines.append(f"  {d['device']:<8} {d['name']}{mem}")
    override = os.environ.get(DEVICE_ENV_VAR)
    lines.append(f"auto -> {resolve_device('auto')}"
                 + (f"   ({DEVICE_ENV_VAR}={override} overrides configs)" if override else ""))
    return "\n".join(lines)
