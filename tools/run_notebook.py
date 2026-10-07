"""Execute a prepared notebook from top to bottom and save the executed copy."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per code cell, in seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient

    source = args.notebook.resolve()
    if source.parent.name != "notebooks":
        parser.error("Canonical notebooks must be run from a prepared notebooks/ directory")
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(source.parent)}})
    client.execute()
    destination = args.output or source.with_name(source.stem + ".executed.ipynb")
    nbformat.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
