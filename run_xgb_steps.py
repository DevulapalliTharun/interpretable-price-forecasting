import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def build_steps(python_exe: str, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return [
        (
            "Fused Retrain with XGBoost (TFT training)",
            [
                python_exe,
                "scripts/10_xgb_as_tft_feature.py",
                "--gpus",
                str(args.gpus),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
            ],
        ),
        (
            "Ensemble Predict step5",
            [
                python_exe,
                "scripts/07_ensemble_predict.py",
                "--family",
                "step5",
                "--gpus",
                str(args.gpus),
            ],
        ),
        (
            "Conformal Calibrate step5",
            [python_exe, "scripts/08_conformal_calibrate.py", "--family", "step5"],
        ),
        ("Explainability Stats", [python_exe, "scripts/11_explainability_stats.py"]),
        ("Final Evaluation", [python_exe, "scripts/06_evaluate.py"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the step5 XGBoost-fused TFT pipeline with the active Python environment."
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request (0=CPU)")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs for step 1 TFT retraining")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Batch size for step 1 TFT retraining")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience for step 1 TFT retraining")
    parser.add_argument("--start-at", type=int, choices=range(1, 6), default=1, help="Start from a later pipeline step")
    parser.add_argument("--log-file", default="xgb_regeneration.log", help="Path to the pipeline log file")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=0,
        help="Optional per-step timeout in hours. Use 0 to disable timeouts.",
    )
    args = parser.parse_args()

    python_exe = sys.executable
    log_path = ROOT / args.log_file
    steps = build_steps(python_exe, args)
    timeout_seconds = None if args.timeout_hours <= 0 else int(args.timeout_hours * 3600)
    completed: list[str] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Starting XGBoost artifact regeneration at {datetime.now().isoformat()}\n")
        log_file.write(f"Python executable: {python_exe}\n")
        log_file.write(f"Working directory: {ROOT}\n")
        log_file.write(f"Requested GPUs: {args.gpus}\n")
        log_file.write(f"Step 1 epochs: {args.epochs}\n")
        log_file.write(f"Step 1 batch size: {args.batch_size}\n")
        log_file.write(f"Step 1 patience: {args.patience}\n\n")

        for index, (name, command) in enumerate(steps, start=1):
            if index < args.start_at:
                completed.append(f"{index}. {name} (skipped)")
                continue

            banner = f"\n{'=' * 70}\nSTEP {index}/5: {name}\nStart time: {datetime.now().isoformat()}\n{'=' * 70}\n"
            print(banner, end="")
            sys.stdout.flush()
            log_file.write(banner)
            log_file.write(f"Command: {' '.join(command)}\n")
            log_file.flush()

            try:
                result = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                message = f"TIMEOUT: Step {index} exceeded {args.timeout_hours} hours"
                print(message)
                log_file.write(message + "\n")
                log_file.write(f"Completed before timeout: {completed}\n")
                return 1
            except Exception as exc:
                message = f"ERROR: {type(exc).__name__}: {exc}"
                print(message)
                log_file.write(message + "\n")
                log_file.write(f"Completed before error: {completed}\n")
                return 1

            if result.returncode != 0:
                message = f"FAILED at step {index}: {name}"
                print(message)
                log_file.write("\n" + message + "\n")
                log_file.write(f"Return code: {result.returncode}\n")
                log_file.write(f"Completed before failure: {completed}\n")
                return result.returncode or 1

            completed.append(f"{index}. {name}")
            success = f"STEP {index} COMPLETED SUCCESSFULLY"
            print(success)
            log_file.write("\n" + success + f" at {datetime.now().isoformat()}\n")
            log_file.flush()

        print(f"\n{'=' * 70}")
        print("SUCCESS: ALL 5 STEPS COMPLETED")
        print("=" * 70)
        for step in completed:
            print(step)
        log_file.write(f"\n{'=' * 70}\n")
        log_file.write("SUCCESS: ALL 5 STEPS COMPLETED\n")
        log_file.write("=" * 70 + "\n")
        for step in completed:
            log_file.write(step + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
