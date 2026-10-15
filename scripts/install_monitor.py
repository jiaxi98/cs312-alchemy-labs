"""Install a durable 2-minute CS312 monitor without modifying other cron entries."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

from cluster import ROOT

MARKER = "# cs312-a1-monitor-80gpu"


def cron_text(previous, line):
    # Preserve every unrelated entry including infrastructure/user monitors.
    retained = [entry for entry in previous.splitlines() if MARKER not in entry]
    return "\n".join([*retained, line]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True,
                        help="Persistent interpreter with requirements-monitor.txt installed")
    args = parser.parse_args()
    interpreter = args.python.absolute()
    subprocess.run([str(interpreter), "-c", "import paramiko, requests"], check=True)
    state = ROOT / "runtime/monitor"
    state.mkdir(parents=True, exist_ok=True)
    if not (state / "stage.local.json").exists():
        raise RuntimeError("Pin the tested deployment in runtime/monitor/stage.local.json first")
    wrapper = state / "cron.sh"
    wrapper.write_text("#!/bin/sh\nset -eu\n"
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "export PYTHONUNBUFFERED=1\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"exec /usr/bin/timeout --signal=TERM --kill-after=10s 110s {shlex.quote(str(interpreter))} "
        f"{shlex.quote(str(ROOT / 'scripts/monitor.py'))} --apply\n")
    wrapper.chmod(0o700)
    line = f"*/2 * * * * /bin/sh {shlex.quote(str(wrapper))} >> {shlex.quote(str(state / 'cron.log'))} 2>&1 {MARKER}"
    previous = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if previous.returncode != 0 and "no crontab" not in previous.stderr.lower():
        raise RuntimeError("Cannot read existing crontab")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    (state / f"crontab-before-{timestamp}.txt").write_text(previous.stdout)
    proposed = cron_text(previous.stdout, line)
    subprocess.run(["crontab", "-"], input=proposed, text=True, check=True)
    readback = subprocess.check_output(["crontab", "-l"], text=True)
    if readback != proposed:
        raise RuntimeError("Crontab readback differs from installed configuration")
    receipt = dict(installed_at_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   interval_seconds=120, cron_entry=line, python=str(interpreter),
                   status="installed", allocation_limit=80)
    (state / "install.local.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
