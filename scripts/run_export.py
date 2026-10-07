"""Run a project exporter with the selected external model checkout on PYTHONPATH."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    "llamagen": {
        "reconstruction": "reconstruct_imagenet.py",
        "usage": "llamagen_code_usage_export.py",
        "relations": "llamagen_precompute_codebook_relations.py",
        "robustness": "llamagen_robustness_experiment_dataset_export.py",
        "reference": "create_val_ref.py",
    },
    "vqgan": {
        "reconstruction": "reconstruct_imagenet_single.py",
        "usage": "vqgan_code_usage_export.py",
        "relations": "vqgan_precompute_codebook_relations.py",
        "robustness": "vqgan_robustness_experiment_dataset_export.py",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=RUNNERS)
    parser.add_argument("--task", required=True, choices=sorted(set().union(*(x.keys() for x in RUNNERS.values()))))
    parser.add_argument("--upstream", type=Path, help="Defaults to third_party/MODEL")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.task not in RUNNERS[args.model]:
        parser.error(f"{args.model} has no {args.task} runner")
    upstream = (args.upstream or ROOT / "third_party" / args.model).resolve()
    marker = "tokenizer/tokenizer_image/vq_model.py" if args.model == "llamagen" else "taming/models/vqgan.py"
    if not (upstream / marker).is_file():
        parser.error(f"Missing upstream model source: {upstream / marker}. See docs/setup.md")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts" / args.model / RUNNERS[args.model][args.task]), *forwarded]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=upstream, env=env, check=True)


if __name__ == "__main__":
    main()
