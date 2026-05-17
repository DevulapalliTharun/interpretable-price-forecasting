#!/usr/bin/env python
"""
Check the current status of the XGBoost artifact regeneration process.
Run this script anytime to see current progress.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "xgb_regeneration.log"

def check_status():
    if not LOG_FILE.exists():
        print("Log file not found.")
        print("Start the pipeline with:")
        print(r"  .\.venv\Scripts\python.exe .\run_xgb_steps.py --gpus 1")
        return

    with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    lines = content.split('\n')
    
    print("=" * 70)
    print("XGBoost Artifact Regeneration - Status")
    print("=" * 70)
    
    # Check for completion/failure first
    if "SUCCESS: ALL 5 STEPS COMPLETED" in content:
        print("\n>>> ALL 5 STEPS COMPLETED SUCCESSFULLY <<<\n")
        # Print the summary
        for i, line in enumerate(lines):
            if "SUCCESS: ALL 5 STEPS COMPLETED" in line:
                for j in range(max(0, i-2), min(len(lines), i+10)):
                    if lines[j].strip():
                        print(lines[j])
        return
    
    if "FAILED at step" in content or "FAILURE:" in content:
        print("\n>>> PROCESS FAILED <<<\n")
        for line in lines:
            if "FAILED" in line or "FAILURE:" in line or "Return code" in line:
                print(line)
        return
    
    if "TIMEOUT:" in content:
        print("\n>>> PROCESS TIMEOUT <<<\n")
        for line in lines:
            if "TIMEOUT" in line:
                print(line)
        return
    
    # Process is still running
    print("\nStatus: RUNNING")
    print(f"Log file: {LOG_FILE}")
    print(f"Log size: {len(content):,} bytes")
    
    # Find current step
    step_markers = [l for l in lines if re.match(r'STEP \d+/5:', l)]
    if step_markers:
        current_step = step_markers[-1]
        print(f"Current: {current_step}")
    
    # Find progress
    progress_lines = [l for l in lines if 'Epoch' in l or '|' in l and '[' in l]
    if progress_lines:
        last = progress_lines[-1]
        match = re.search(r'(\d+)%', last)
        if match:
            pct = match.group(1)
            print(f"Progress: {pct}%")
        
        # Try to estimate time remaining
        if int(pct) < 100:
            # Rough estimate based on pattern
            elapsed_epoch_lines = len([l for l in progress_lines if 'Epoch 0' in l or 'Epoch 1' in l])
            print(f"Activity: {elapsed_epoch_lines} training updates logged")
    
    completed = content.count("COMPLETED SUCCESSFULLY")
    print(f"Completed steps: {completed}/5")
    
    print("\n(Run this script again to check progress)")
    print("=" * 70)

if __name__ == "__main__":
    check_status()
