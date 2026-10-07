"""Create a non-destructive NAS index; never move datasets or replace existing paths."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "configs/nas-catalog.json")
    parser.add_argument("--nas-root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Without this flag only check targets")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    root = (args.nas_root or Path(catalog["nas_root"])).resolve()
    destination = root / catalog["handover_directory"]
    missing = []
    for name, target in catalog["links"].items():
        source = Path(target)
        if not source.is_absolute():
            source = root / source
        if not source.exists():
            missing.append(str(source))
            continue
        link = destination / name
        if link.is_symlink() and link.resolve() == source.resolve():
            print(f"OK {name}")
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace {link}")
        if args.apply:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source, target_is_directory=source.is_dir())
        print(f"{'LINK' if args.apply else 'CHECK'} {name} -> {source}")
    if missing:
        print("Missing targets:\n" + "\n".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
