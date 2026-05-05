from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, stream_command, timestamp, write_json
from prepare_mnist_png import parse_digits, prepare_mnist_png


DEFAULT_COMMIT = "5cc59a486ec1f122403d324e8882bec810440624"
DEFAULT_CLONE_URL = "https://github.com/matthieuheitz/WassersteinDictionaryLearning.git"


LOSS_RE = re.compile(r"loss:\s*([-+0-9.eE]+)\s+step:\s*([-+0-9.eE]+)")
ITER_RE = re.compile(r"(?:LBFGS\s+)?Iteration\s+(\d+)(?:,\s+total iterations\s+(\d+))?")
FINAL_TIME_RE = re.compile(r"time taken \(s\) :\s*([-+0-9.eE]+)")


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, check=True)


def clone_or_reuse_source(source_dir: Path, clone_url: str, commit: str) -> str:
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", clone_url, source_dir])
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"{source_dir} exists but is not a git checkout")
    run_checked(["git", "fetch", "--depth", "1", "origin", commit], cwd=source_dir)
    run_checked(["git", "checkout", "--detach", commit], cwd=source_dir)
    resolved = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(source_dir), text=True
    ).strip()
    return resolved


def apply_noavx_fallback(source_dir: Path) -> bool:
    helper_path = source_dir / "cpp" / "sse_helpers.h"
    text = helper_path.read_text()
    marker = "Scalar fallback used when AVX dotp_full is unavailable."
    if marker in text:
        return False
    needle = "typedef __m128d simd_double;\ntypedef __m128 simd_float;\n#endif\n\n#endif\n"
    replacement = (
        "typedef __m128d simd_double;\n"
        "typedef __m128 simd_float;\n\n"
        "// Scalar fallback used when AVX dotp_full is unavailable.\n"
        "static inline double dotp_full(const double* u, const double* v, int n) {\n"
        "\tdouble conv = 0;\n"
        "\tfor (int k = 0; k < n; k++) {\n"
        "\t\tconv += u[k] * v[k];\n"
        "\t}\n"
        "\treturn conv;\n"
        "}\n"
        "#endif\n\n"
        "#endif\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find insertion point in {helper_path}")
    helper_path.write_text(text.replace(needle, replacement))
    return True


def default_avx_mode() -> str:
    # Apple Silicon has no AVX; Rosetta reports x86_64 but sysctl still says
    # AVX is unavailable.
    if platform.system() == "Darwin":
        try:
            avx = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.avx1_0"], text=True
            ).strip()
            return "on" if avx == "1" else "off"
        except Exception:
            return "off"
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return "off"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(errors="ignore").lower()
        if " avx " not in f" {text} ":
            return "off"
    return "on"


def build_binary(source_dir: Path, build_dir: Path, *, avx_mode: str, with_openmp: bool) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    with_avx = avx_mode == "on"
    if not with_avx:
        apply_noavx_fallback(source_dir)

    cmake_cmd = [
        "cmake",
        f"-DWITH_OPENMP={'ON' if with_openmp else 'OFF'}",
        "-DWITH_HALIDE=OFF",
        "-DWITH_EIGEN=OFF",
        f"-DWITH_AVX_SUPPORT={'ON' if with_avx else 'OFF'}",
        "-DBUILD_APP_DICTIONARY_LEARNING=ON",
        source_dir / "cpp",
    ]
    run_checked(cmake_cmd, cwd=build_dir)
    run_checked(["cmake", "--build", ".", "-j"], cwd=build_dir)
    binary = build_dir / "app_dictionary_learning"
    if not binary.exists():
        raise RuntimeError(f"Build finished but binary is missing: {binary}")
    return binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Heitz/Schmitz WDL C++ baseline.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "experiments" / "results" / f"heitz_{timestamp()}")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--clone-url", default=DEFAULT_CLONE_URL)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of same-sized PNG histograms. If omitted, MNIST PNGs are generated.")
    parser.add_argument("--mnist-root", type=Path, default=REPO_ROOT / "mnist_raw")
    parser.add_argument("--max-per-digit", type=int, default=10)
    parser.add_argument("--digits", nargs="*", default=None)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--avx", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--with-openmp", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--loss-type", type=int, default=2)
    parser.add_argument("--sinkhorn-iters", type=int, default=10)
    parser.add_argument("--scale-dict-factor", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-optim-iter", type=int, default=10)
    parser.add_argument("--export-every", type=int, default=0)
    parser.add_argument("--warm-restart", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--im-complement", action="store_true")
    parser.add_argument("--allow-neg-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    source_dir = args.source_dir or (run_dir / "external" / "WassersteinDictionaryLearning")
    build_dir = args.build_dir or (run_dir / "build" / "heitz_wdl")
    input_dir = args.input_dir
    data_metadata = None
    if input_dir is None:
        data_dir = run_dir / "data" / "mnist_png"
        data_metadata = prepare_mnist_png(
            data_dir,
            mnist_root=args.mnist_root,
            max_per_digit=args.max_per_digit,
            digits=parse_digits(args.digits),
            force=args.force_data,
        )
        input_dir = Path(data_metadata["image_dir"])

    source_commit = clone_or_reuse_source(source_dir, args.clone_url, args.commit)
    avx_mode = default_avx_mode() if args.avx == "auto" else args.avx
    binary = build_binary(source_dir, build_dir, avx_mode=avx_mode, with_openmp=args.with_openmp)

    output_dir = run_dir / "outputs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()
    log_path = run_dir / "stdout.log"

    eval_index = 0
    iter_index = 0
    final_reported_time = None

    def on_line(line: str, elapsed: float) -> None:
        nonlocal eval_index, iter_index, final_reported_time
        loss_match = LOSS_RE.search(line)
        if loss_match:
            eval_index += 1
            event = {
                "event": "loss_eval",
                "method": "heitz_wdl",
                "eval_index": eval_index,
                "elapsed_seconds": elapsed,
                "loss": float(loss_match.group(1)),
                "line_search_step": float(loss_match.group(2)),
            }
            with history_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        iter_match = ITER_RE.search(line)
        if iter_match:
            iter_index += 1
            event = {
                "event": "lbfgs_iteration",
                "method": "heitz_wdl",
                "iteration_event_index": iter_index,
                "elapsed_seconds": elapsed,
                "lbfgs_iteration": int(iter_match.group(1)),
                "total_iteration": int(iter_match.group(2) or iter_match.group(1)),
            }
            with history_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        final_match = FINAL_TIME_RE.search(line)
        if final_match:
            final_reported_time = float(final_match.group(1))

    cmd = [
        binary,
        "-i", input_dir,
        "-o", output_dir,
        "-k", args.k,
        "-l", args.loss_type,
        "-n", args.sinkhorn_iters,
        "-s", args.scale_dict_factor,
        "-g", args.gamma,
        "-a", args.alpha,
        "-x", args.max_optim_iter,
        "-m", args.export_every,
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.warm_restart:
        cmd.append("--warmRestart")
    if args.im_complement:
        cmd.append("--imComplement")
    if args.allow_neg_weights:
        cmd.append("--allowNegWeights")

    returncode, elapsed = stream_command(cmd, cwd=build_dir, log_path=log_path, on_line=on_line)
    summary = {
        "method": "heitz_wasserstein_dictionary_learning",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "reported_elapsed_seconds": final_reported_time,
        "history_path": str(history_path),
        "stdout_log": str(log_path),
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_dir": str(source_dir),
        "source_commit": source_commit,
        "binary": str(binary),
        "build": {
            "avx": avx_mode,
            "with_openmp": args.with_openmp,
            "platform_machine": platform.machine(),
        },
        "parameters": {
            "k": args.k,
            "loss_type": args.loss_type,
            "sinkhorn_iters": args.sinkhorn_iters,
            "scale_dict_factor": args.scale_dict_factor,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "max_optim_iter": args.max_optim_iter,
            "export_every": args.export_every,
            "warm_restart": args.warm_restart,
            "deterministic": args.deterministic,
            "im_complement": args.im_complement,
            "allow_neg_weights": args.allow_neg_weights,
        },
        "data_metadata": data_metadata,
        "command": [str(part) for part in cmd],
    }
    write_json(run_dir / "summary.json", summary)
    if returncode != 0:
        sys.exit(returncode)


if __name__ == "__main__":
    main()
