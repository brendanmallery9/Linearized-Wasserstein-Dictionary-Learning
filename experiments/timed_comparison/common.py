from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def stream_command(
    cmd: Iterable[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str, float], None] | None = None,
) -> tuple[int, float]:
    """Run a command, teeing stdout/stderr and returning return code + elapsed time."""
    cmd = [str(part) for part in cmd]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    log_file = log_path.open("w") if log_path is not None else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            if on_line is not None:
                on_line(line.rstrip("\n"), time.monotonic() - start)
    finally:
        if log_file is not None:
            log_file.close()

    returncode = proc.wait()
    elapsed = time.monotonic() - start
    return returncode, elapsed

