"""Phase 4: the frozen agent with each belief model, on the evaluation splits.

Variants (``eval.variants``), each a full ``experiments/dra_inference.py``
run per split under ``phase4_eval/{variant}/``:

    off       no belief model
    teacher   the teacher belief model (phase 1) in inform mode
    student   the trained belief model in inform mode, served by a local
              vLLM server (``vllm/belief-student`` at $BELIEF_VLLM_API_BASE)

The criteria come from the same model (and criteria bank) in every variant;
only the belief updater differs.  Inference resume applies, so a finished
variant is not run again.  The scores are the existing evaluators' (the
run's summary.json): answer correctness (QA) or the Auto-ARGUE nugget
metrics (reports), and Recall@N / GradedRecall@N of the seen documents.
They are collected in ``phase4_eval/summary.json``.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RunPaths, Split, TrainingConfig
from .trajectories import REPO_ROOT, find_run_dir, inference_command

STUDENT_NAME = "belief-student"
VARIANTS = ("off", "teacher", "student")


def run_scores(run_dir: Path) -> Dict[str, Any]:
    """Correctness and recall of one run, from its summary.json."""
    summary = json.loads((run_dir / "summary.json").read_text())
    generation = summary.get("generation") or {}
    seen = ((summary.get("retrieval") or {}).get("seen") or {}).get("Metrics@N") or {}
    scores: Dict[str, Any] = {"num_queries": summary.get("num_queries")}
    if "correctness" in generation:
        scores["accuracy"] = generation["correctness"].get("accuracy")
    if "nuggets" in generation:
        scores.update(generation["nuggets"].get("metrics") or {})
    scores["Recall@N"] = seen.get("Recall@N")
    scores["GradedRecall@N"] = seen.get("GradedRecall@N")
    return scores


class StudentServer:
    """``vllm serve`` of the student on one GPU, for the inference runs."""

    def __init__(self, model_path: str, cfg: TrainingConfig, log_file: Path) -> None:
        ec = cfg.eval
        self.base_url = f"http://127.0.0.1:{ec.serve_port}/v1"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_file, "a")
        vllm_bin = shutil.which("vllm") or "vllm"
        self._proc = subprocess.Popen(
            [vllm_bin, "serve", model_path, "--served-model-name", STUDENT_NAME,
             "--port", str(ec.serve_port), "--dtype", "bfloat16",
             "--max-model-len", str(ec.serve_max_model_len),
             "--gpu-memory-utilization", str(ec.serve_gpu_memory_utilization)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(ec.serve_gpu)},
            stdout=self._log, stderr=subprocess.STDOUT,
        )

    def wait(self, timeout: float = 900.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"the student vLLM server exited (code {self._proc.returncode}); see {self._log.name}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5):
                    return
            except OSError:
                time.sleep(5)
        raise TimeoutError(f"the student vLLM server did not start within {timeout:.0f}s; see {self._log.name}")

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._log.close()


def run_eval(cfg: TrainingConfig, paths: RunPaths) -> None:
    ec, tc = cfg.eval, cfg.trajectories
    if not ec.splits:
        raise ValueError("no evaluation splits (eval.splits / --eval-splits)")
    unknown = [v for v in ec.variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown eval variants {unknown}; expected some of {VARIANTS}")
    splits = [Split.parse(s) for s in ec.splits]
    agent = ec.agent or tc.agent
    student = ec.belief_model or str(paths.rl_final)

    server: Optional[StudentServer] = None
    results: Dict[str, Dict[str, Any]] = {}
    try:
        for variant in ec.variants:
            env = dict(os.environ)
            judge = tc.teacher_model
            if variant == "student":
                if not Path(student).exists():
                    raise FileNotFoundError(f"student model {student} not found")
                server = StudentServer(student, cfg, paths.eval_dir / "student_server.log")
                server.wait()
                env["BELIEF_VLLM_API_BASE"] = server.base_url
                judge = f"vllm/{STUDENT_NAME}"
            mode = "off" if variant == "off" else "inform"
            output = paths.eval_dir / variant
            for split in splits:
                cmd = inference_command(
                    cfg, split, agent=agent, belief_mode=mode, output=output,
                    belief_judge_model=judge, criteria_model=tc.criteria_model,
                    gpu_ids=ec.inference_gpu_ids,
                )
                print(f"[phase 4] {variant} {split.name}: {' '.join(cmd)}", flush=True)
                subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
                run_dir = find_run_dir(output, split, cfg.shared.retriever, mode)
                results.setdefault(split.name, {})[variant] = {"run_dir": str(run_dir), **run_scores(run_dir)}
            if server is not None:
                server.stop()
                server = None
    finally:
        if server is not None:
            server.stop()

    summary = {"agent": agent, "teacher_model": tc.teacher_model, "student_model": student,
               "criteria_model": tc.criteria_model, "splits": results}
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_summary.write_text(json.dumps(summary, indent=2))
    for split_name, by_variant in results.items():
        print(f"[phase 4] {split_name}")
        for variant, scores in by_variant.items():
            shown = {k: v for k, v in scores.items() if k != "run_dir"}
            print(f"    {variant:8s} {json.dumps(shown)}")
    print(f"[phase 4] summary saved to {paths.eval_summary}")
