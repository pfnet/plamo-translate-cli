"""Reject tracked benchmark files other than reviewed README summaries."""

import subprocess
import sys
from pathlib import PurePosixPath


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z", "--", "benchmarks/"])
    unexpected = [
        path.decode("utf-8", errors="replace")
        for path in paths.split(b"\0")
        if path and PurePosixPath(path.decode("utf-8", errors="replace")).name != "README.md"
    ]
    if unexpected:
        print("Only README.md summaries may be tracked under benchmarks/:", file=sys.stderr)
        for path in unexpected:
            print(f"  {path}", file=sys.stderr)
        return 1
    print("Benchmark content policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
