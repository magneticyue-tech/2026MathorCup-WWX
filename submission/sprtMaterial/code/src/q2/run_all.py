# code/src/q2/run_all.py
"""Q2 正式运行：按依赖顺序逐步执行、检查返回码、保存完整控制台日志与 run_manifest.json。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config as C
from data import input_checks

STEPS = [
    "data.py", "graph.py", "association.py", "inference.py", "stability.py", "rationality.py",
    "controls_sensitivity.py", "full_controls.py", "simulation.py", "sensitivity_runs.py",
    "check_attachment6.py", "advanced_audits.py", "extra_figures.py", "pipeline_final.py", "acceptance.py",
]
INPUTS = [C.CHEM_PATH, C.CORR_PATH, C.SENSORY_PATH, C.ATT6_PATH, C.ATT8_PATH, C.Q1_DATA_NPZ,
          C.Q1_MAIN_OOF_NPZ, C.CHEMICAL_DICT_PATH, C.DISCORDANCE_PATH]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(C.PROJECT_ROOT).as_posix()


def main(start: str | None):
    log_path = C.Q2_DIR / "run_log.txt"
    steps = STEPS[STEPS.index(start):] if start else STEPS
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.PROJECT_ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        head = ""
    records = []
    old = C.Q2_DIR / "run_manifest.json"
    if start and old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))["steps"]
        done = [rel(CURRENT_DIR / s) for s in STEPS[: STEPS.index(start)]]
        records = [r for r in prev if r["command"].split()[-1] in done]
    t_all = time.time()
    mode = "a" if start else "w"
    with open(log_path, mode, encoding="utf-8") as log:
        log.write(f"# Q2 正式运行 {datetime.now().isoformat(timespec='seconds')}  HEAD {head}\n")
        for s in steps:
            cmd = [sys.executable, rel(CURRENT_DIR / s)]
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            print(f"$ {' '.join(cmd)}", flush=True)
            t0 = time.time()
            p = subprocess.Popen(cmd, cwd=C.PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            rc = p.wait()
            records.append({"command": " ".join(cmd), "returncode": rc, "seconds": round(time.time() - t0, 1)})
            log.write(f"[返回码 {rc}，{records[-1]['seconds']}s]\n")
            if rc != 0:
                print(f"步骤 {s} 失败（返回码 {rc}），停止下游步骤")
                break
    sens_meta = {p.stem.replace("_meta", ""): json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(C.SENS_DIR.glob("*_meta.json"))}
    outputs = {}
    for root in (C.Q2_DIR, C.FIGURES):
        for p in sorted(root.rglob("*")):
            if p.is_file() and "_smoke" not in p.parts and p.name != "run_manifest.json":
                outputs[rel(p)] = sha256(p)
    manifest = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(time.time() - t_all, 1),
        "git_head": head,
        "python": sys.executable,
        "inputs": {rel(p): sha256(p) for p in INPUTS},
        "input_checks": input_checks(),
        "steps": records,
        "seed": C.SEED,
        "n_bootstrap": C.N_BOOTSTRAP,
        "n_permutation": C.N_PERMUTATION,
        "n_simulation": C.N_SIMULATION,
        "tau_main": C.TAU_MAIN,
        "m3": {"outer_splits": f"{C.M3_N_SPLITS}x{C.M3_N_REPEATS_SPLIT}", "perm_repeats": C.M3_PERM_REPEATS},
        "sensitivity": sens_meta,
        "outputs_sha256": outputs,
        "console_log": rel(log_path),
    }
    (C.Q2_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(r["returncode"] != 0 for r in records):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=STEPS)
    main(ap.parse_args().start)
