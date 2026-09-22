"""Runtime configuration. The only platform-aware code in the project.

Called once, before any worker thread starts. Two of these settings are not
optional:

  set_num_threads(1)         32 workers x 12 intra-op threads oversubscribes a
                             12-core machine by 32x. It is also a source of
                             nondeterminism, and this project asserts bitwise
                             reproducibility.
  set_num_interop_threads(1) Must be called before any parallel work or torch
                             raises. Hence the try/except.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvInfo:
    python: str
    torch: str
    platform: str
    machine: str
    cpu_count: int
    is_colab: bool
    source_sha256: str

    def render(self) -> str:
        return "\n".join(
            [
                "provenance",
                f"  python         {self.python}",
                f"  torch          {self.torch}",
                f"  platform       {self.platform} ({self.machine})",
                f"  cpu_count      {self.cpu_count}",
                f"  colab          {self.is_colab}",
                f"  vzero sha256   {self.source_sha256[:16]}",
            ]
        )


def _is_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content")


def source_sha256() -> str:
    """Hash of the vzero sources, so a committed notebook says which engine
    produced its numbers."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as fh:
                h.update(name.encode())
                h.update(fh.read())
    return h.hexdigest()


def configure(seed: int = 0) -> EnvInfo:
    import torch

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started parallel work; harmless on a re-run
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    return EnvInfo(
        python=sys.version.split()[0],
        torch=torch.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        is_colab=_is_colab(),
        source_sha256=source_sha256(),
    )
