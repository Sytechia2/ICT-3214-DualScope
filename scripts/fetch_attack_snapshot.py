#!/usr/bin/env python3
"""Download a pinned Enterprise ATT&CK release and write the technique catalogue (Task 6.1).

Fetches the STIX 2.1 bundle from MITRE's attack-stix-data repository, records
its SHA-256 and the retrieval date, and writes the compact catalogue that
retrieval and verification load. The bundle itself (about 54 MB) stays in
``data/raw/attack/`` and is not committed; the catalogue is.

Pass ``--bundle`` to rebuild from an already downloaded file; its SHA-256 must
match the one recorded in the existing catalogue unless ``--version`` changes.

Example:
  python scripts/fetch_attack_snapshot.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.attack.catalog import collection_info, extract_techniques  # noqa: E402

URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack-{version}.json"
DEFAULT_VERSION = "19.2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--bundle", type=Path, help="Use this downloaded bundle instead of fetching it")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/attack")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/reference/attack/enterprise_techniques.json")
    args = parser.parse_args()

    url = URL.format(version=args.version)
    if args.bundle is None:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        args.bundle = args.raw_dir / f"enterprise-attack-{args.version}.json"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, args.bundle)
        retrieved = time.strftime("%Y-%m-%d", time.gmtime())
    else:
        retrieved = time.strftime("%Y-%m-%d", time.gmtime(args.bundle.stat().st_mtime))
    raw = args.bundle.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))["snapshot"]
        if previous["attack_version"] == args.version and previous["bundle_sha256"] != digest:
            raise SystemExit(f"bundle for ATT&CK {args.version} differs from the recorded one ({previous['bundle_sha256']})")

    bundle = json.loads(raw)
    info = collection_info(bundle)
    if info["version"] != args.version:
        raise SystemExit(f"bundle is ATT&CK {info['version']}, expected {args.version}")
    techniques = extract_techniques(bundle)
    snapshot = {
        "collection": info["name"],
        "attack_version": info["version"],
        "collection_modified": info["modified"],
        "source_url": url,
        "bundle_sha256": digest,
        "retrieved_utc_date": retrieved,
        "techniques": len(techniques),
        "subtechniques": sum(t["is_subtechnique"] for t in techniques),
        "filter": "attack-pattern objects that are not revoked or deprecated",
        "licence": "ATT&CK is copyright The MITRE Corporation; used under the ATT&CK Terms of Use (https://attack.mitre.org/resources/legal-and-branding/terms-of-use/)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"snapshot": snapshot, "techniques": techniques}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
