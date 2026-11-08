"""
Validate requirements files: malformed lines, and version drift between files.

Two checks:

1. Every line must parse. pip-audit and pip stop at the FIRST bad line; this
   does not -- it parses each requirement independently, so one run shows the
   whole mess instead of making you fix them one round-trip at a time.

2. Given two or more files, any package pinned in more than one file must be
   pinned to the SAME version. Independently-compiled lock files drift apart
   silently, and a package left clean in one file but vulnerable in the other
   is exactly the gap an audit run can walk straight past.

Usage:
    python check_requirements.py requirements.txt requirements-dev.txt

Exit status: 0 = clean, 1 = at least one problem (so it can gate CI).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("need `packaging`:  pip install packaging")

# Lines starting with these are pip options, not requirements.
OPTION_PREFIXES = ("-", "--")


def logical_lines(path: Path):
    """Yield (first_lineno, text) with backslash continuations joined."""
    buf: list[str] = []
    start = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        if not buf:
            start = lineno
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        buf.append(line)
        yield start, "".join(buf).strip()
        buf = []
    if buf:
        yield start, "".join(buf).strip()


def strip_hashes(text: str) -> str:
    """Drop --hash=... fragments that pip-compile appends."""
    return text.split("--hash=")[0].strip()


def parse(path: Path):
    """Return (bad_lines, pins) for one file.

    bad_lines: [(lineno, text, reason)]
    pins:      {canonical_name: (lineno, version_or_None)}
    """
    bad: list[tuple[int, str, str]] = []
    pins: dict[str, tuple[int, str | None]] = {}

    for lineno, text in logical_lines(path):
        if not text or text.startswith("#"):
            continue
        # Strip a trailing inline comment, but not a '#' inside a URL fragment.
        if " #" in text:
            text = text.split(" #", 1)[0].strip()
        if text.startswith(OPTION_PREFIXES):
            continue  # -r / -c / --index-url / --extra-index-url / --find-links
        candidate = strip_hashes(text)
        if not candidate:
            continue
        try:
            req = Requirement(candidate)
        except InvalidRequirement as exc:
            bad.append((lineno, candidate, str(exc).splitlines()[0]))
            continue

        name = canonicalize_name(req.name)
        exact = [s.version for s in req.specifier if s.operator in ("==", "===")]
        pins[name] = (lineno, exact[0] if exact else None)

    return bad, pins


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit(__doc__)

    problems = 0
    all_pins: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)

    for name in argv:
        path = Path(name)
        if not path.is_file():
            print(f"!! {name}: not found")
            problems += 1
            continue

        bad, pins = parse(path)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())

        if bad:
            print(f"\n{path}  ({n_lines} lines) -- {len(bad)} unparseable line(s):")
            for lineno, text, why in bad:
                print(f"  line {lineno}: {text!r}")
                print(f"           -> {why}")
            problems += len(bad)
        else:
            print(f"OK  {path}  ({n_lines} lines, {len(pins)} requirements, all parse)")

        for pkg, (lineno, version) in pins.items():
            all_pins[pkg].append((str(path), lineno, version))

    # Cross-file version drift.
    drift = {pkg: places for pkg, places in all_pins.items() if len(places) > 1 and len({v for _, _, v in places}) > 1}
    if drift:
        print(f"\nVERSION DRIFT -- {len(drift)} package(s) pinned differently across files:")
        for pkg in sorted(drift):
            print(f"  {pkg}:")
            for fname, lineno, version in drift[pkg]:
                shown = version if version is not None else "(not an exact pin)"
                print(f"    {fname}:{lineno}  {shown}")
        problems += len(drift)
    elif len(argv) > 1:
        shared = sum(1 for places in all_pins.values() if len(places) > 1)
        print(f"OK  no version drift ({shared} package(s) appear in more than one file, all agreeing)")

    if problems:
        print(f"\n{problems} problem(s) total.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
