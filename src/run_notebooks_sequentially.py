from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
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
        description="Executa notebooks em sequencia usando jupyter nbconvert."
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=[
            "src/load_dataframes.ipynb",
            "src/bert_fake_real_workbench.ipynb",
        ],
        help=(
            "Lista de notebooks a executar em ordem. "
            "Padrao: src/load_dataframes.ipynb src/bert_fake_real_workbench.ipynb"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/executed_notebooks",
        help="Pasta onde os notebooks executados serao salvos.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
        help="Timeout por celula em segundos para o ExecutePreprocessor.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continua para o proximo notebook mesmo se houver falha.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Executa e salva no proprio arquivo .ipynb (sem usar output-dir).",
    )
    return parser.parse_args()


def run_notebook(
    project_root: Path,
    notebook_relative: str,
    output_dir: Path,
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
            error_message="Arquivo nao encontrado.",
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
    process = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
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
    message = stderr or stdout or "Falha sem mensagem retornada pelo nbconvert."
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
    output_dir = (project_root / args.output_dir).resolve()

    results: list[NotebookRunResult] = []

    print("Executando notebooks em sequencia...")
    for notebook in args.notebooks:
        print(f"\n[RUN] {notebook}")
        result = run_notebook(
            project_root=project_root,
            notebook_relative=notebook,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
            inplace=args.inplace,
        )
        results.append(result)

        if result.status == "ok":
            print(
                f"[OK] {result.notebook.name} em {result.duration_seconds:.1f}s"
                f" -> {result.output_file}"
            )
            continue

        print(f"[ERROR] {result.notebook} ({result.error_message})")
        if not args.continue_on_error:
            print("Parando execucao por falha. Use --continue-on-error para seguir.")
            break

    print("\nResumo:")
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
