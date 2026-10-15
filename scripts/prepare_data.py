"""Download the pinned official dataset, checking every manifest SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def download_ranges(url, target, entry, workers):
    """Parallel, bounded requests into disjoint file offsets; verify the whole file."""
    size = entry["size_bytes"]
    chunk_size = 32 * 1024**2
    count = (size + chunk_size - 1) // chunk_size
    partial = target.with_suffix(target.suffix + ".ranges")
    record = target.with_suffix(target.suffix + ".ranges.json")
    identity = dict(url=url, size=size, sha256=entry["sha256"], chunk_size=chunk_size)
    completed = set()
    if record.exists() and partial.exists():
        previous = json.loads(record.read_text())
        if previous["identity"] != identity or partial.stat().st_size != size:
            raise RuntimeError("Existing partial download has a different identity")
        completed = set(previous["completed"])
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, size)

    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor the byte range")
                    offset = start
                    while chunk := response.read(min(4 * 1024**2, end - offset + 2)):
                        if offset + len(chunk) > end + 1:
                            raise RuntimeError("Range response exceeds expected length")
                        view = memoryview(chunk)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            if not written:
                                raise OSError("Zero-length write")
                            view = view[written:]
                            offset += written
                    if offset != end + 1:
                        raise OSError("Truncated byte range")
                return index
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
        raise RuntimeError("Range failed")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(fetch, i) for i in range(count) if i not in completed]
            for future in as_completed(pending):
                completed.add(future.result())
                temp = record.with_suffix(".tmp")
                temp.write_text(json.dumps({"identity": identity, "completed": sorted(completed)}))
                temp.replace(record)
                if len(completed) % 16 == 0 or len(completed) == count:
                    print(f"{target.name}: {len(completed)}/{count} ranges", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = sha256(partial)
    if digest != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {target.name}")
    partial.replace(target)
    record.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text())["dataset"]
    base = f"https://huggingface.co/datasets/{lock['repo_id']}/resolve/{lock['revision']}/"
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + "manifest.json", timeout=30) as response:
        raw = response.read()
    manifest = json.loads(raw)
    (root / "manifest.json").write_bytes(raw)
    start = time.monotonic()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe dataset manifest path")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == entry["size_bytes"]:
            digest = sha256(target)
            if digest == entry["sha256"]:
                print("Verified existing", relative, flush=True)
                continue
        if entry["size_bytes"] >= 256 * 1024**2 and args.workers > 1:
            download_ranges(base + entry["path"], target, entry, args.workers)
            print("Verified", relative, entry["sha256"], flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        # The official server supports byte ranges. Resume only with Content-Range
        # matching the exact existing offset; otherwise start this file again.
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(base + entry["path"],
                headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Incorrect range response")
                    else:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as output:
                        last_report = offset
                        print("Downloading", relative, "from", offset, flush=True)
                        while chunk := response.read(8 * 1024**2):
                            output.write(chunk)
                            offset += len(chunk)
                            if offset - last_report >= 512 * 1024**2:
                                print(f"{relative}: {offset}/{entry['size_bytes']} bytes", flush=True)
                                last_report = offset
                break
            except (OSError, TimeoutError) as exc:
                if attempt == 2:
                    raise
                print("Download interrupted; will resume:", type(exc).__name__, flush=True)
        if partial.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Incorrect size: {relative}")
        digest = sha256(partial)
        if digest != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        partial.replace(target)
        print("Verified", relative, digest, flush=True)
    (root / "READY.json").write_text(json.dumps({
        **lock, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "files": manifest["files"], "seconds": time.monotonic() - start,
    }, indent=2) + "\n")
    print("Data ready", root, flush=True)


if __name__ == "__main__":
    main()
