from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


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

    return proc.wait(), time.monotonic() - start


DEFAULT_COMMIT = "5cc59a486ec1f122403d324e8882bec810440624"
DEFAULT_CLONE_URL = "https://github.com/matthieuheitz/WassersteinDictionaryLearning.git"


FLOAT_RE = r"[-+0-9.eE]+|[-+]?nan|[-+]?inf"
LOSS_RE = re.compile(rf"loss:\s*({FLOAT_RE})\s+step:\s*({FLOAT_RE})")
ITER_RE = re.compile(r"(?:LBFGS\s+)?Iteration\s+(\d+)(?:,\s+total iterations\s+(\d+))?")
FINAL_TIME_RE = re.compile(r"time taken \(s\) :\s*([-+0-9.eE]+)")
EARLY_STOP_RE = re.compile(
    rf"early_stop reason=([A-Za-z0-9_]+) iteration=(\d+) loss=({FLOAT_RE}) elapsed_seconds=({FLOAT_RE})"
)


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


def apply_quiet_logging_patch(source_dir: Path) -> bool:
    """Reduce Heitz runtime output to losses and accepted iterations."""
    changed = False
    main_path = source_dir / "cpp" / "main_dictionary_learning.cpp"
    inverse_path = source_dir / "cpp" / "inverseWasserstein.h"

    main_text = main_path.read_text()
    init_dump = (
        '\t// Display initialization\n'
        '\tstd::cout<<"Initialization (just weights+20): "<<std::endl;\n'
        '\tfor (int i=0; i<K*P+20; i++) {\n'
        '\t\tstd::cout<<variables[i]<<" ";\n'
        '\t}\n'
        '\tstd::cout<<std::endl;\n'
    )
    init_quiet = (
        '\t// Display initialization\n'
        '\tstd::cout<<"Initialization complete: samples="<<P<<" atoms="<<K<<" variables="<<variables.size()<<std::endl;\n'
    )
    if init_dump in main_text:
        main_text = main_text.replace(init_dump, init_quiet)
        changed = True

    solution_dump = (
        '\t// Display final solution\n'
        '\tstd::cout<<"solution (just weights+20): "<<std::endl;\n'
        '\tfor (int i=0; i<K*P+20; i++) {\n'
        '\t\tstd::cout<<variables[i]<<" ";\n'
        '\t}\n'
        '\tstd::cout<<std::endl;\n'
    )
    solution_quiet = (
        '\t// Display final solution\n'
        '\tstd::cout<<"solution ready"<<std::endl;\n'
    )
    if solution_dump in main_text:
        main_text = main_text.replace(solution_dump, solution_quiet)
        changed = True
    main_path.write_text(main_text)

    inverse_text = inverse_path.read_text()
    bar_print = '\t\t\tstd::cout<<"|"<<std::flush;\n'
    if bar_print in inverse_text:
        inverse_text = inverse_text.replace(bar_print, "")
        changed = True

    loss_print = (
        '\t\tstd::cout<<std::endl;\n'
        '\t\tstd::cout<<"loss: "<<lossVal<<" \\tstep: "<<step<<std::endl;\n'
    )
    loss_quiet = (
        '\t\tstd::cout<<"loss_eval iteration="<<regression->iteration<<" loss: "<<lossVal<<" \\tstep: "<<step<<std::endl;\n'
    )
    if loss_print in inverse_text:
        inverse_text = inverse_text.replace(loss_print, loss_quiet)
        changed = True

    new_iter_print = '\t\tstd::cout<<"new iter-----------------"<<std::endl;\n'
    if new_iter_print in inverse_text:
        inverse_text = inverse_text.replace(new_iter_print, "")
        changed = True

    progress_dump = (
        '\t\tif(regression->warmRestart)\n'
        '\t\t\tprintf("LBFGS Iteration %d, total iterations %d :\\n", k,regression->iteration);\n'
        '\t\telse\n'
        '\t\t\tprintf("Iteration %d:\\n", k);\n'
        '\t\tprintf("time elapsed: %f (s)\\n", regression->chrono.GetDiffMs()*0.001);\n'
    )
    progress_quiet = (
        '\t\tif(regression->warmRestart)\n'
        '\t\t\tprintf("LBFGS Iteration %d, total iterations %d, loss: %f, elapsed_seconds: %f\\n", k,regression->iteration, fx, regression->chrono.GetDiffMs()*0.001);\n'
        '\t\telse\n'
        '\t\t\tprintf("Iteration %d, loss: %f, elapsed_seconds: %f\\n", k, fx, regression->chrono.GetDiffMs()*0.001);\n'
    )
    if progress_dump in inverse_text:
        inverse_text = inverse_text.replace(progress_dump, progress_quiet)
        changed = True

    weights_dump_start = '\t\t// Display fitting variables\n'
    weights_dump_end = '\t\treturn 0;\n'
    if weights_dump_start in inverse_text:
        start = inverse_text.index(weights_dump_start)
        end = inverse_text.index(weights_dump_end, start)
        inverse_text = inverse_text[:start] + "\t\treturn 0;\n" + inverse_text[end + len(weights_dump_end):]
        changed = True

    inverse_path.write_text(inverse_text)
    return changed


def apply_linux_cxx_link_patch(source_dir: Path) -> bool:
    """Make the mixed C/C++ Heitz target link reliably on Linux clusters."""
    cmake_path = source_dir / "cpp" / "CMakeLists.txt"
    text = cmake_path.read_text()
    marker = "Codex Linux CXX link patch"
    if marker in text and "CODEX_LIBSTDCXX" in text:
        return False

    needle = "    target_link_libraries(app_dictionary_learning ${ADDITIONAL_LINKER_FLAG})\n"
    patch_block = (
        f"    # {marker}: keep mixed C/C++ target on the C++ linker and\n"
        "    # explicitly link the libstdc++ that belongs to CMAKE_CXX_COMPILER.\n"
        "    set_target_properties(app_dictionary_learning PROPERTIES LINKER_LANGUAGE CXX)\n"
        "    if(CMAKE_SYSTEM_NAME STREQUAL \"Linux\")\n"
        "        execute_process(\n"
        "            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so\n"
        "            OUTPUT_VARIABLE CODEX_LIBSTDCXX\n"
        "            OUTPUT_STRIP_TRAILING_WHITESPACE\n"
        "        )\n"
        "        if(CODEX_LIBSTDCXX AND EXISTS \"${CODEX_LIBSTDCXX}\")\n"
        "            target_link_libraries(app_dictionary_learning \"${CODEX_LIBSTDCXX}\")\n"
        "        else()\n"
        "            target_link_libraries(app_dictionary_learning stdc++)\n"
        "        endif()\n"
        "    endif()\n"
    )
    if marker in text:
        old_block = (
            f"    # {marker}: keep mixed C/C++ target on the C++ linker and\n"
            "    # explicitly add libstdc++ on Linux toolchains that omit it.\n"
            "    set_target_properties(app_dictionary_learning PROPERTIES LINKER_LANGUAGE CXX)\n"
            "    if(CMAKE_SYSTEM_NAME STREQUAL \"Linux\")\n"
            "        target_link_libraries(app_dictionary_learning stdc++)\n"
            "    endif()\n"
        )
        if old_block not in text:
            raise RuntimeError(f"Could not upgrade existing Linux CXX link patch in {cmake_path}")
        cmake_path.write_text(text.replace(old_block, patch_block))
        return True

    replacement = needle + patch_block
    if needle not in text:
        raise RuntimeError(f"Could not find app link target in {cmake_path}")
    cmake_path.write_text(text.replace(needle, replacement))
    return True


def apply_early_stopping_patch(source_dir: Path) -> bool:
    """Add clean LBFGS stopping criteria to the cloned Heitz baseline."""
    changed = False
    main_path = source_dir / "cpp" / "main_dictionary_learning.cpp"
    inverse_path = source_dir / "cpp" / "inverseWasserstein.h"

    main_text = main_path.read_text()
    marker = "Codex early stopping patch"
    if marker not in main_text:
        usage_needle = (
            '\t\t\t\t\t\t"[-m <exportEveryMIter>] "\n'
            '\t\t\t\t\t\t"[--deterministic] "\n'
        )
        usage_replacement = (
            '\t\t\t\t\t\t"[-m <exportEveryMIter>] "\n'
            '\t\t\t\t\t\t"[--maxElapsedSeconds <seconds>] "\n'
            '\t\t\t\t\t\t"[--plateauWindow <n>] "\n'
            '\t\t\t\t\t\t"[--plateauMinDelta <delta>] "\n'
            '\t\t\t\t\t\t"[--deterministic] "\n'
        )
        if usage_needle not in main_text:
            raise RuntimeError(f"Could not find usage insertion point in {main_path}")
        main_text = main_text.replace(usage_needle, usage_replacement)

        defaults_needle = (
            "\tint maxOptimIter = 200;\n"
            "\tint exportEveryMIter = 0;\n"
            "\tbool warmRestart = false;\n"
        )
        defaults_replacement = (
            "\tint maxOptimIter = 200;\n"
            "\tint exportEveryMIter = 0;\n"
            "\t// Codex early stopping patch: disabled when values are <= 0.\n"
            "\tdouble maxElapsedSeconds = 0.0;\n"
            "\tint plateauWindow = 0;\n"
            "\tdouble plateauMinDelta = 0.0;\n"
            "\tbool warmRestart = false;\n"
        )
        if defaults_needle not in main_text:
            raise RuntimeError(f"Could not find defaults insertion point in {main_path}")
        main_text = main_text.replace(defaults_needle, defaults_replacement)

        parse_needle = (
            '\t\t} else if (std::string(argv[i]) == "-m") {\n'
            '\t\t\texportEveryMIter = std::stoi(argv[i + 1]);\n'
            '\t\t} else if (std::string(argv[i]) == "--deterministic") {\n'
        )
        parse_replacement = (
            '\t\t} else if (std::string(argv[i]) == "-m") {\n'
            '\t\t\texportEveryMIter = std::stoi(argv[i + 1]);\n'
            '\t\t} else if (std::string(argv[i]) == "--maxElapsedSeconds") {\n'
            '\t\t\tmaxElapsedSeconds = std::stof(argv[i + 1]);\n'
            '\t\t} else if (std::string(argv[i]) == "--plateauWindow") {\n'
            '\t\t\tplateauWindow = std::stoi(argv[i + 1]);\n'
            '\t\t} else if (std::string(argv[i]) == "--plateauMinDelta") {\n'
            '\t\t\tplateauMinDelta = std::stof(argv[i + 1]);\n'
            '\t\t} else if (std::string(argv[i]) == "--deterministic") {\n'
        )
        if parse_needle not in main_text:
            raise RuntimeError(f"Could not find parser insertion point in {main_path}")
        main_text = main_text.replace(parse_needle, parse_replacement)

        config_needle = (
            "\tregression.exportEveryMIter = exportEveryMIter;\n"
            "\tregression.exp_weight = !allowNegWeights;\n"
            "\tregression.wrTotalIteration = maxOptimIter;\n"
        )
        config_replacement = (
            "\tregression.exportEveryMIter = exportEveryMIter;\n"
            "\tregression.exp_weight = !allowNegWeights;\n"
            "\tregression.wrTotalIteration = maxOptimIter;\n"
            "\tregression.maxElapsedSeconds = maxElapsedSeconds;\n"
            "\tregression.plateauWindow = plateauWindow;\n"
            "\tregression.plateauMinDelta = plateauMinDelta;\n"
        )
        if config_needle not in main_text:
            raise RuntimeError(f"Could not find regression config insertion point in {main_path}")
        main_text = main_text.replace(config_needle, config_replacement)
        main_path.write_text(main_text)
        changed = True

    inverse_text = inverse_path.read_text()
    if "if(fx != fx)" in inverse_text:
        inverse_text = inverse_text.replace(
            "if(fx != fx)",
            "if(fx != fx || fx > 1e308 || fx < -1e308)",
        )
        inverse_path.write_text(inverse_text)
        changed = True
    if marker not in inverse_text:
        constructor_needle = (
            "\t\texportOnlyFinalSolution = export_only_final_solution;\n"
            "\t\twarmRestart = warm_restart;\n"
            "\t\texportEveryMIter = 1;\n"
            "\t\tlbfgs_parameter_init(&lbfgs_param);\n"
        )
        constructor_replacement = (
            "\t\texportOnlyFinalSolution = export_only_final_solution;\n"
            "\t\twarmRestart = warm_restart;\n"
            "\t\texportEveryMIter = 1;\n"
            "\t\t// Codex early stopping patch: disabled by default.\n"
            "\t\tmaxElapsedSeconds = 0.0;\n"
            "\t\tplateauWindow = 0;\n"
            "\t\tplateauMinDelta = 0.0;\n"
            "\t\tearlyStopRequested = false;\n"
            "\t\tearlyStopReason = \"\";\n"
            "\t\tbestPlateauAverage = 0.0;\n"
            "\t\thasBestPlateauAverage = false;\n"
            "\t\tlbfgs_parameter_init(&lbfgs_param);\n"
        )
        if constructor_needle not in inverse_text:
            raise RuntimeError(f"Could not find constructor insertion point in {inverse_path}")
        inverse_text = inverse_text.replace(constructor_needle, constructor_replacement)

        loop_needle = (
            "\t\t\tif(this->warmRestart)\n"
            "\t\t\t{\n"
            "\t\t\t\t// Store the last computed scalings\n"
            "\t\t\t\tstd::copy(this->b_temp.begin(), this->b_temp.end(),this->b_storage.begin());\n"
            "\t\t\t}\n"
        )
        loop_replacement = (
            "\t\t\tif(this->warmRestart)\n"
            "\t\t\t{\n"
            "\t\t\t\t// Store the last computed scalings\n"
            "\t\t\t\tstd::copy(this->b_temp.begin(), this->b_temp.end(),this->b_storage.begin());\n"
            "\t\t\t}\n"
            "\t\t\tif(this->earlyStopRequested)\n"
            "\t\t\t{\n"
            "\t\t\t\tbreak;\n"
            "\t\t\t}\n"
        )
        if loop_needle not in inverse_text:
            raise RuntimeError(f"Could not find LBFGS loop insertion point in {inverse_path}")
        inverse_text = inverse_text.replace(loop_needle, loop_replacement)

        progress_needle = (
            "\t\tif(regression->warmRestart)\n"
            "\t\t\tprintf(\"LBFGS Iteration %d, total iterations %d :\\n\", k,regression->iteration);\n"
            "\t\telse\n"
            "\t\t\tprintf(\"Iteration %d:\\n\", k);\n"
            "\t\tprintf(\"time elapsed: %f (s)\\n\", regression->chrono.GetDiffMs()*0.001);\n"
        )
        progress_replacement = (
            "\t\tdouble elapsedSeconds = regression->chrono.GetDiffMs()*0.001;\n"
            "\t\tif(regression->warmRestart)\n"
            "\t\t\tprintf(\"LBFGS Iteration %d, total iterations %d :\\n\", k,regression->iteration);\n"
            "\t\telse\n"
            "\t\t\tprintf(\"Iteration %d:\\n\", k);\n"
            "\t\tprintf(\"time elapsed: %f (s)\\n\", elapsedSeconds);\n"
            "\n"
            "\t\tif(fx != fx || fx > 1e308 || fx < -1e308)\n"
            "\t\t{\n"
            "\t\t\tregression->earlyStopRequested = true;\n"
            "\t\t\tregression->earlyStopReason = \"nonfinite_loss\";\n"
            "\t\t}\n"
            "\t\tif(!regression->earlyStopRequested && regression->maxElapsedSeconds > 0.0 && elapsedSeconds >= regression->maxElapsedSeconds)\n"
            "\t\t{\n"
            "\t\t\tregression->earlyStopRequested = true;\n"
            "\t\t\tregression->earlyStopReason = \"max_elapsed_seconds\";\n"
            "\t\t}\n"
            "\t\tif(!regression->earlyStopRequested && regression->plateauWindow > 0)\n"
            "\t\t{\n"
            "\t\t\tregression->plateauLosses.push_back(fx);\n"
            "\t\t\tif(regression->plateauLosses.size() >= regression->plateauWindow)\n"
            "\t\t\t{\n"
            "\t\t\t\tdouble sumLoss = 0.0;\n"
            "\t\t\t\tfor(int idx = regression->plateauLosses.size() - regression->plateauWindow; idx < regression->plateauLosses.size(); ++idx)\n"
            "\t\t\t\t{\n"
            "\t\t\t\t\tsumLoss += regression->plateauLosses[idx];\n"
            "\t\t\t\t}\n"
            "\t\t\t\tdouble avgLoss = sumLoss / regression->plateauWindow;\n"
            "\t\t\t\tif(!regression->hasBestPlateauAverage)\n"
            "\t\t\t\t{\n"
            "\t\t\t\t\tregression->bestPlateauAverage = avgLoss;\n"
            "\t\t\t\t\tregression->hasBestPlateauAverage = true;\n"
            "\t\t\t\t}\n"
            "\t\t\t\telse if(regression->bestPlateauAverage - avgLoss > regression->plateauMinDelta)\n"
            "\t\t\t\t{\n"
            "\t\t\t\t\tregression->bestPlateauAverage = avgLoss;\n"
            "\t\t\t\t}\n"
            "\t\t\t\telse\n"
            "\t\t\t\t{\n"
            "\t\t\t\t\tregression->earlyStopRequested = true;\n"
            "\t\t\t\t\tregression->earlyStopReason = \"loss_plateau\";\n"
            "\t\t\t\t}\n"
            "\t\t\t}\n"
            "\t\t}\n"
            "\t\tif(regression->earlyStopRequested)\n"
            "\t\t{\n"
            "\t\t\tprintf(\"early_stop reason=%s iteration=%d loss=%f elapsed_seconds=%f\\n\", regression->earlyStopReason.c_str(), regression->iteration, fx, elapsedSeconds);\n"
            "\t\t\treturn LBFGSERR_CANCELED;\n"
            "\t\t}\n"
        )
        if progress_needle not in inverse_text:
            raise RuntimeError(f"Could not find progress insertion point in {inverse_path}")
        inverse_text = inverse_text.replace(progress_needle, progress_replacement)

        fields_needle = (
            "\t// For warm restart\n"
            "\tbool warmRestart;\n"
            "\tint wrTotalIteration;\n"
            "};\n"
        )
        fields_replacement = (
            "\t// For warm restart\n"
            "\tbool warmRestart;\n"
            "\tint wrTotalIteration;\n"
            "\t// Codex early stopping patch\n"
            "\tdouble maxElapsedSeconds;\n"
            "\tint plateauWindow;\n"
            "\tdouble plateauMinDelta;\n"
            "\tbool earlyStopRequested;\n"
            "\tstd::string earlyStopReason;\n"
            "\tstd::vector<double> plateauLosses;\n"
            "\tdouble bestPlateauAverage;\n"
            "\tbool hasBestPlateauAverage;\n"
            "};\n"
        )
        if fields_needle not in inverse_text:
            raise RuntimeError(f"Could not find field insertion point in {inverse_path}")
        inverse_text = inverse_text.replace(fields_needle, fields_replacement)
        inverse_path.write_text(inverse_text)
        changed = True

    return changed


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
    parser.add_argument("--max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=0)
    parser.add_argument("--plateau-min-delta", type=float, default=0.0)
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
        from prepare_mnist_png import parse_digits, prepare_mnist_png

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
    apply_early_stopping_patch(source_dir)
    apply_quiet_logging_patch(source_dir)
    apply_linux_cxx_link_patch(source_dir)
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
    termination_reason = None
    termination_iteration = None
    termination_loss = None
    termination_elapsed_seconds = None

    def on_line(line: str, elapsed: float) -> None:
        nonlocal eval_index, iter_index, final_reported_time
        nonlocal termination_reason, termination_iteration, termination_loss, termination_elapsed_seconds
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
        early_stop_match = EARLY_STOP_RE.search(line)
        if early_stop_match:
            termination_reason = early_stop_match.group(1)
            termination_iteration = int(early_stop_match.group(2))
            termination_loss = float(early_stop_match.group(3))
            termination_elapsed_seconds = float(early_stop_match.group(4))
            with history_path.open("a") as f:
                f.write(json.dumps({
                    "event": "termination",
                    "method": "heitz_wdl",
                    "reason": termination_reason,
                    "iteration": termination_iteration,
                    "elapsed_seconds": termination_elapsed_seconds,
                    "loss": termination_loss,
                }) + "\n")

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
    if args.max_elapsed_seconds is not None:
        cmd.extend(["--maxElapsedSeconds", args.max_elapsed_seconds])
    if args.plateau_window:
        cmd.extend(["--plateauWindow", args.plateau_window])
        cmd.extend(["--plateauMinDelta", args.plateau_min_delta])

    returncode, elapsed = stream_command(cmd, cwd=build_dir, log_path=log_path, on_line=on_line)
    if termination_reason is None:
        if iter_index >= args.max_optim_iter:
            termination_reason = "max_optim_iter"
        else:
            termination_reason = "completed"
    if termination_iteration is None:
        termination_iteration = iter_index
    if termination_elapsed_seconds is None:
        termination_elapsed_seconds = elapsed
    summary = {
        "method": "heitz_wasserstein_dictionary_learning",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "reported_elapsed_seconds": final_reported_time,
        "termination_reason": termination_reason,
        "termination_iteration": termination_iteration,
        "termination_loss": termination_loss,
        "termination_elapsed_seconds": termination_elapsed_seconds,
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
            "max_elapsed_seconds": args.max_elapsed_seconds,
            "plateau_window": args.plateau_window,
            "plateau_min_delta": args.plateau_min_delta,
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
