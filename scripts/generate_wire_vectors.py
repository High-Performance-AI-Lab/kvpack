#!/usr/bin/env python3
"""Emit the digest-pinned production-v1 deterministic vector manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference" / "python"))

import kvpack_v1_ref as kv  # noqa: E402
import production_v1 as cli  # noqa: E402


def record(suite: str, name: str, value: bytes) -> str:
    value = bytes(value)
    return f"{suite}\t{name}\t{len(value)}\t{hashlib.sha256(value).hexdigest()}"


def main() -> None:
    print("# kvpack production-v1 mutable-wire deterministic vectors")
    print("# suite\tname\tbytes\tsha256")
    for codec_name, codec in (("raw", kv.CODEC_RAW), ("lossless", kv.CODEC_LOSSLESS)):
        for chunk_mode in ("plain", "encrypted"):
            for pack_mode in ("plain", "encrypted"):
                suite = f"fixture/{codec_name}/{chunk_mode}/{pack_mode}"
                data = cli.fixture_json(codec_name, chunk_mode, pack_mode)
                for name in (
                    "state_declaration",
                    "family",
                    "schema",
                    "manifest",
                    "raw_frame",
                    "lossless_frame",
                    "chunk",
                    "pack",
                    "semantic_id",
                    "family_id",
                    "schema_id",
                    "manifest_id",
                    "namespace_id",
                    "token_root",
                    "auxiliary_root",
                    "chunk_id",
                    "object_key",
                    "object_digest",
                ):
                    value = data[name]
                    assert isinstance(value, str)
                    print(record(suite, name, bytes.fromhex(value)))
                keys = data["keys"]
                assert isinstance(keys, dict)
                for name in sorted(keys):
                    value = keys[name]
                    assert isinstance(value, str)
                    print(record(suite, f"key/{name}", bytes.fromhex(value)))

                chain = kv.reference_delta_chain(
                    codec,
                    chunk_mode == "encrypted",
                    pack_mode == "encrypted",
                )
                chain_suite = f"delta-chain/{codec_name}/{chunk_mode}/{pack_mode}"
                for stage, fixture in enumerate(chain):
                    print(record(chain_suite, f"stage-{stage}/manifest", kv.encode_manifest(fixture.manifest)))
                    print(record(chain_suite, f"stage-{stage}/chunk", fixture.chunk.data))
                    print(record(chain_suite, f"stage-{stage}/pack", fixture.pack))
                    print(record(chain_suite, f"stage-{stage}/manifest-id", fixture.pack_id))
                    print(record(chain_suite, f"stage-{stage}/plaintext", fixture.plaintext))


if __name__ == "__main__":
    main()
