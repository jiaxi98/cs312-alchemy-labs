"""Store a SwanLab API key using hidden terminal input, outside the repository."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    key = getpass.getpass("SwanLab API key (hidden): ").strip()
    if not key:
        raise SystemExit("No key supplied; nothing changed.")
    workspace = input("SwanLab workspace (Enter = personal account): ").strip() or None
    if workspace and not re.fullmatch(r"[0-9a-zA-Z_-]+", workspace):
        raise SystemExit("Workspace must be the account ID (for example jiaxi98), not the display name.")
    project = input("SwanLab project [cs312-alchemy-labs]: ").strip() or "cs312-alchemy-labs"
    directory = Path.home() / ".config/cs312"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    key_file = directory / "swanlab-key"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(key + "\n")
    settings = dict(project=project, workspace=workspace, api_key_file=str(key_file))
    output = ROOT / "reports/swanlab.local.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(settings, indent=2) + "\n")
    print("SwanLab credentials configured. Key hidden and excluded from Git.")


if __name__ == "__main__":
    main()
