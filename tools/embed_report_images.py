"""Mechanically embed local PNG image sources so the HTML can be shared alone."""
import argparse
import base64
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    def replace(match):
        source = match.group(2)
        if source.startswith("data:"):
            return match.group(0)
        if source.startswith(("http:", "https:")):
            raise ValueError("Do not download or embed remote image resources")
        path = args.html.parent / source
        if path.suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG: {path}")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return match.group(1) + "data:image/png;base64," + data + match.group(3)
    content = re.sub(r'(<img[^>]+src=")([^"]+)(")', replace, args.html.read_text())
    args.html.write_text(content)
    print(f"Embedded image sources in {args.html} ({args.html.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
