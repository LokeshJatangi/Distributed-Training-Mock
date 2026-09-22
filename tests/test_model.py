
"""Model initialization must be reproducible across Python processes."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    "from vzero.model import ModelConfig, init_param; "
    "print(init_param('head.w', (8, 8), ModelConfig()).numpy().tobytes().hex())"
)


def test_init_param_ignores_python_hash_seed():
    outputs = []
    for hash_seed in ("0", "12345"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(subprocess.check_output(
            [sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, text=True
        ))
    assert outputs[0] == outputs[1]
