from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class NotebookRunResult:
    notebook: Path
    status: str
    duration_seconds: float
    output_file: Path | None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs notebooks sequentially using jupyter nbconvert."
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=[
            "src/llm_simulation_workbench.ipynb",
            "src/bert_fake_real_workbench.ipynb",
            "src/pretrained_fake_news_detector_workbench.ipynb",
        ],
        help=(
            "List of notebooks to run in order. "
            "Default: src/llm_simulation_workbench.ipynb src/bert_fake_real_workbench.ipynb "
            "src/pretrained_fake_news_detector_workbench.ipynb"
        ),
    )
    parser.add_argument(
        "--runs-root",
        default="output/runs",
        help=(
            "Root directory for runs. Each run creates output/runs/<run_id>/... "
            "with executed notebooks and notebook artifacts."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier. If omitted, uses a UTC timestamp "
            "(ex.: 20260320_184500)."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Per-cell timeout in seconds for ExecutePreprocessor.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next notebook even if one fails.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Execute and save in the same .ipynb file (without using output-dir).",
    )
    return parser.parse_args()


def run_notebook(
    project_root: Path,
    notebook_relative: str,
    output_dir: Path,
    run_id: str,
    run_dir: Path,
    timeout_seconds: int,
    inplace: bool,
) -> NotebookRunResult:
    notebook_path = (project_root / notebook_relative).resolve()
    if not notebook_path.exists():
        return NotebookRunResult(
            notebook=notebook_path,
            status="not_found",
            duration_seconds=0.0,
            output_file=None,
            error_message="File not found.",
        )

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        f"--ExecutePreprocessor.timeout={timeout_seconds}",
    ]

    output_file: Path | None
    if inplace:
        cmd.append("--inplace")
        output_file = notebook_path
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output", notebook_path.name, "--output-dir", str(output_dir)])
        output_file = output_dir / notebook_path.name

    start = time.perf_counter()
    env = dict(os.environ)
    env["RUN_ID"] = run_id
    env["RUN_DIR"] = str(run_dir)

    process = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - start

    if process.returncode == 0:
        return NotebookRunResult(
            notebook=notebook_path,
            status="ok",
            duration_seconds=elapsed,
            output_file=output_file,
        )

    stderr = (process.stderr or "").strip()
    stdout = (process.stdout or "").strip()
    message = stderr or stdout or "Failure with no message returned by nbconvert."
    return NotebookRunResult(
        notebook=notebook_path,
        status="error",
        duration_seconds=elapsed,
        output_file=output_file,
        error_message=message,
    )


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / args.runs_root / run_id).resolve()
    output_dir = run_dir / "executed_notebooks"

    results: list[NotebookRunResult] = []

    print("Running notebooks sequentially...")
    print(f"RUN_ID: {run_id}")
    print(f"RUN_DIR: {run_dir}")
    for notebook in args.notebooks:
        print(f"\n[RUN] {notebook}")
        result = run_notebook(
            project_root=project_root,
            notebook_relative=notebook,
            output_dir=output_dir,
            run_id=run_id,
            run_dir=run_dir,
            timeout_seconds=args.timeout_seconds,
            inplace=args.inplace,
        )
        results.append(result)

        if result.status == "ok":
            print(
                f"[OK] {result.notebook.name} in {result.duration_seconds:.1f}s"
                f" -> {result.output_file}"
            )
            continue

        print(f"[ERROR] {result.notebook} ({result.error_message})")
        if not args.continue_on_error:
            print("Stopping execution due to failure. Use --continue-on-error to continue.")
            break

    print("\nSummary:")
    for item in results:
        msg = (
            f"- {item.notebook.name}: {item.status} "
            f"({item.duration_seconds:.1f}s)"
        )
        if item.output_file:
            msg += f" -> {item.output_file}"
        print(msg)

    has_error = any(item.status != "ok" for item in results)
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
