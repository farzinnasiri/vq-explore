"""Fetch recorded upstream source revisions into ignored third_party checkouts."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "configs/upstreams.lock.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=lock)
    parser.add_argument("--root", type=Path, default=ROOT / "third_party")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for model in args.models:
        spec = lock[model]
        dest = args.root.resolve() / model
        if dest.exists():
            head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"], text=True).strip()
            if head != spec["commit"] or dirty:
                raise RuntimeError(f"Existing checkout differs from the recorded clean revision: {dest}")
            print(f"Already present: {dest}")
            continue
        commands = [
            ["git", "init", str(dest)],
            ["git", "-C", str(dest), "remote", "add", "origin", spec["url"]],
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", spec["commit"]],
            ["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"],
        ]
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
