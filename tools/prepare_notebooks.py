"""Create small, portable copies of existing analysis notebooks."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SETUP = '''import os
from pathlib import Path

# Set these before launching Jupyter when using another storage mount.
VQ_NAS_ROOT = Path(os.environ.get("VQ_NAS_ROOT", "/megaverse/storage/nasiri"))
VQ_JUPYTER_ROOT = Path(os.environ.get("VQ_JUPYTER_ROOT", str(VQ_NAS_ROOT / "jupyter")))
'''


def prepare(source, destination, exploratory=False):
    notebook = json.loads(source.read_text())
    for cell in notebook["cells"]:
        cell.pop("attachments", None)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            text = "".join(cell["source"])
            # Only replace absolute /exp Path constructors and getenv defaults.
            text = re.sub(
                r'Path\(([\"\x27])/exp/?([^\"\x27]*)\1\)',
                lambda m: 'VQ_JUPYTER_ROOT' if not m[2] else f'VQ_JUPYTER_ROOT / "{m[2]}"',
                text,
            )
            text = re.sub(
                r'(os\.environ\.get\([^,]+,\s*)([\"\x27])/exp/([^\"\x27]*)\2',
                lambda m: m[1] + f'str(VQ_JUPYTER_ROOT / "{m[3]}")',
                text,
            )
            text = text.replace('os.environ.get("CH08_METADATA_BASE", "/mnt/jupyter")',
                                'os.environ.get("CH08_METADATA_BASE", str(VQ_JUPYTER_ROOT))')
            cell["source"] = text.splitlines(keepends=True)
    notebook["metadata"].pop("widgets", None)
    notebook["metadata"].pop("signature", None)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"
    }
    title = "Exploratory notebook" if exploratory else "Reproduction notebook"
    notebook["cells"][:0] = [
        {"cell_type": "markdown", "metadata": {}, "id": "handover-instructions",
         "source": [f"## {title} setup\n", "\n",
                    "See `docs/notebooks.md` for inputs, execution order, and environment setup.\n",
                    "Run a working copy prepared by `tools/prepare_workspace.py`.\n",
                    "Saved outputs were removed from this copy; executed originals remain on the NAS.\n",
                    "The following cell changes file locations only. The analysis formulas are preserved.\n"]},
        {"cell_type": "code", "metadata": {}, "id": "handover-storage",
         "execution_count": None, "outputs": [], "source": SETUP.splitlines(keepends=True)},
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    print(prepare(args.source, args.destination, args.exploratory))


if __name__ == "__main__":
    main()
