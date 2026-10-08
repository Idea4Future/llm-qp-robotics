"""Fetch the pinned official KaTeX browser files for offline/local serving."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

VERSION = "0.16.22"
ARCHIVE_URL = f"https://github.com/KaTeX/KaTeX/releases/download/v{VERSION}/katex.tar.gz"
LICENSE_URL = f"https://raw.githubusercontent.com/KaTeX/KaTeX/v{VERSION}/LICENSE"
DESTINATION = Path(__file__).resolve().parent / "static" / "vendor" / "katex"


def ready():
    return all((DESTINATION / name).is_file() for name in ("katex.min.js", "katex.min.css", "LICENSE", "manifest.json"))


def fetch():
    """Copy only built CSS/JS/fonts, rejecting archive paths outside that set."""
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
        archive = response.read()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for item in bundle.getmembers():
            if not item.isfile() or not item.name.startswith("katex/"):
                continue
            relative = Path(item.name).relative_to("katex")
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("Unsafe KaTeX archive path")
            if relative.as_posix() not in ("katex.min.js", "katex.min.css") and not (
                len(relative.parts) == 2 and relative.parts[0] == "fonts" and relative.suffix in (".woff2", ".woff", ".ttf")
            ):
                continue
            stream = bundle.extractfile(item)
            if stream is None:
                raise ValueError(f"Missing archive payload: {item.name}")
            files[relative.as_posix()] = stream.read()
    if not {"katex.min.js", "katex.min.css"}.issubset(files) or not any(n.endswith(".woff2") for n in files):
        raise ValueError("KaTeX release did not contain the required browser files")
    with urllib.request.urlopen(LICENSE_URL, timeout=30) as response:
        files["LICENSE"] = response.read()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = DESTINATION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {"version": VERSION, "archive_url": ARCHIVE_URL, "license_url": LICENSE_URL,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}}
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    report = fetch()
    print(json.dumps({"version": report["version"], "files": len(report["files"]), "destination": str(DESTINATION)}))
