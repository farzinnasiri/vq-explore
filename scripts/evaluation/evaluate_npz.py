"""Run the separately installed OpenAI evaluator on a reference and sample NPZ."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parents[2] / "third_party/guided_diffusion")
    parser.add_argument("--output-dir", type=Path, required=True, help="Inception download/cache and evaluator log location")
    args = parser.parse_args()
    evaluator = args.upstream.resolve() / "evaluations/evaluator.py"
    if not evaluator.is_file():
        parser.error(f"Install the external evaluator first: {evaluator}")
    for path in (args.reference, args.samples):
        if not path.is_file():
            parser.error(f"Missing NPZ: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluator.log").open("w") as log:
        subprocess.run([sys.executable, str(evaluator), str(args.reference.resolve()), str(args.samples.resolve())],
                       cwd=args.output_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(args.output_dir / "evaluator.log")


if __name__ == "__main__":
    main()
