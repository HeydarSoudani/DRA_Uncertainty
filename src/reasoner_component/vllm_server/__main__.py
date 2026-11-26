"""CLI: run a registered vLLM server in the foreground.

    python -m reasoner_component.vllm_server list
    python -m reasoner_component.vllm_server serve <model> [--port N] [--tp N]
        [--max-model-len N] [--max-num-seqs N] [--dry-run] [any vllm flag ...]

The ``vllm_server_scripts/*.sh`` wrappers call it.
"""

import argparse
import dataclasses
import os
import shlex
from typing import List, Optional

from .process import fit_tp, per_gpu_gb, server_command, server_env, visible_gpus, vllm_version
from .registry import SERVERS, VLLMServerError, require_spec


_TP_FLAGS = ("--tensor-parallel-size", "-tp")


def main(argv: Optional[List[str]] = None) -> None:
    # No abbreviations: a vllm flag passed through must not be taken for a
    # prefix of one of ours (``--max-model`` for ``--max-model-len``).
    parser = argparse.ArgumentParser(prog="python -m reasoner_component.vllm_server", allow_abbrev=False,
                                     description="Serve a registered vLLM model in the foreground.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_cmd = commands.add_parser("serve", allow_abbrev=False,
                                    help="run the server of MODEL (unknown flags go to vllm)")
    serve_cmd.add_argument("model", help="a registered model name (see `list`)")
    serve_cmd.add_argument("--port", type=int)
    serve_cmd.add_argument("--tp", type=int, help="tensor parallel size (default: fitted to the GPUs)")
    serve_cmd.add_argument("--max-model-len", type=int)
    serve_cmd.add_argument("--max-num-seqs", type=int)
    serve_cmd.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    commands.add_parser("list", help="the registered servers")
    args, extra = parser.parse_known_args(argv)

    if args.command == "list":
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        for name, spec in sorted(SERVERS.items(), key=lambda item: (item[1].role, item[1].port, item[0])):
            print(f"{spec.role:<9} {spec.port:<5} {name}")
        return

    # TP decides the GPUs the server gets, so it is set with --tp only.
    if any(a in _TP_FLAGS or a.split("=", 1)[0] in _TP_FLAGS for a in extra):
        parser.error("set the tensor parallel size with --tp (it also picks the GPUs)")

    try:
        spec = require_spec(args.model)
    except VLLMServerError as e:
        parser.error(str(e))
    overrides = {"port": args.port, "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs}
    spec = dataclasses.replace(spec, **{k: v for k, v in overrides.items() if v is not None},
                               extra_args=(*spec.extra_args, *extra))

    visible = visible_gpus()
    tp = args.tp or fit_tp(spec, per_gpu_gb(), max_tp=len(visible) if visible else max(spec.allowed_tp))
    if visible and tp > len(visible):
        parser.error(f"TP={tp} needs {tp} GPUs, {len(visible)} visible ({visible})")
    gpus = visible[:tp] if visible else [str(i) for i in range(tp)]

    cmd = server_command(spec, tp)
    print(f"[vLLM] {spec.label}: model {spec.name}, port {spec.port}, TP {tp}, GPUs {','.join(gpus)}, "
          f"vLLM {vllm_version()}")
    if args.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={','.join(gpus)} {shlex.join(cmd)}")
        return
    os.execve(cmd[0], cmd, server_env(gpus))


if __name__ == "__main__":
    main()
