#!/usr/bin/env python3
"""CLI for the independent stdlib-only production-v1 Python reference."""

from __future__ import annotations

import json
import struct
import sys

import kvpack_v1_ref as kv


def schema_vector() -> bytes:
    return kv.encode_state_declaration(
        kv.StateDeclaration(
            kv.StateKey(7, "attention.k"),
            kv.Shape((320, 4, 8)),
            kv.Shape((64, 4, 8)),
            (32, 8, 1),
            256,
            64,
            320,
            0,
            9,
        )
    )


def fixture_json(codec_name: str, chunk_mode: str, pack_mode: str) -> dict[str, object]:
    codecs = {"raw": kv.CODEC_RAW, "lossless": kv.CODEC_LOSSLESS}
    if codec_name not in codecs or chunk_mode not in {"plain", "encrypted"} or pack_mode not in {
        "plain",
        "encrypted",
    }:
        raise SystemExit("fixture-json CODEC(raw|lossless) CHUNK_MODE PACK_MODE")
    fixture = kv.reference_fixture(
        codecs[codec_name],
        chunk_mode == "encrypted",
        pack_mode == "encrypted",
    )
    manifest = fixture.manifest
    _, nodes = kv.derive_input_cut(
        fixture.keys.prefix,
        manifest.tenant_namespace,
        manifest.semantic_model,
        manifest.family,
        fixture.tokens,
        fixture.auxiliary_inputs,
    )
    return {
        "codec": codec_name,
        "chunk_mode": chunk_mode,
        "pack_mode": pack_mode,
        "state_declaration": schema_vector().hex(),
        "family": kv.encode_family(manifest.family).hex(),
        "schema": kv.encode_schema(manifest.realized_schema).hex(),
        "manifest": kv.encode_manifest(manifest).hex(),
        "raw_frame": kv.encode_codec_frame(kv.CODEC_RAW, fixture.plaintext).hex(),
        "lossless_frame": kv.encode_codec_frame(kv.CODEC_LOSSLESS, fixture.plaintext).hex(),
        "chunk": fixture.chunk.data.hex(),
        "pack": fixture.pack.hex(),
        "semantic_id": kv.semantic_model_id(manifest.semantic_model).hex(),
        "family_id": kv.representation_family_id(manifest.family).hex(),
        "schema_id": kv.realized_schema_id(manifest.realized_schema).hex(),
        "manifest_id": fixture.pack_id.hex(),
        "namespace_id": kv.namespace_id(fixture.keys.namespace, b"tenant-alpha").hex(),
        "token_root": manifest.input_cut.token_root.hex(),
        "auxiliary_root": manifest.input_cut.auxiliary_input_root.hex(),
        "prefix_nodes": [
            {"count": node.token_count, "id": node.node_id.hex(), "reusable": node.reusable}
            for node in nodes
        ],
        "chunk_id": fixture.chunk.chunk_id.hex(),
        "object_key": fixture.chunk.object_key.hex(),
        "object_digest": fixture.chunk.object_digest.hex(),
        "keys": {
            "namespace": fixture.keys.namespace.hex(),
            "prefix": fixture.keys.prefix.hex(),
            "manifest_auth": fixture.keys.manifest_auth.hex(),
            "manifest_encryption": fixture.keys.manifest_encryption.hex(),
            "chunk_identity": fixture.keys.chunk_identity.hex(),
            "object_identity": fixture.keys.object_identity.hex(),
            "chunk_encryption": fixture.keys.chunk_encryption.hex(),
        },
    }


def fixture_lines(codec_name: str, chunk_mode: str, pack_mode: str) -> str:
    data = fixture_json(codec_name, chunk_mode, pack_mode)
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, str):
            lines.append(f"{key}\t{value}")
        elif key == "keys":
            assert isinstance(value, dict)
            for name, encoded in value.items():
                lines.append(f"keys.{name}\t{encoded}")
        elif key == "prefix_nodes":
            assert isinstance(value, list)
            for index, node in enumerate(value):
                lines.append(f"prefix.{index}.count\t{node['count']}")
                lines.append(f"prefix.{index}.id\t{node['id']}")
                lines.append(f"prefix.{index}.reusable\t{int(node['reusable'])}")
    return "\n".join(lines)


def delta_chain_lines(codec_name: str, chunk_mode: str, pack_mode: str) -> str:
    codecs = {"raw": kv.CODEC_RAW, "lossless": kv.CODEC_LOSSLESS}
    if codec_name not in codecs or chunk_mode not in {"plain", "encrypted"} or pack_mode not in {
        "plain",
        "encrypted",
    }:
        raise SystemExit("delta-chain-lines CODEC(raw|lossless) CHUNK_MODE PACK_MODE")
    chain = kv.reference_delta_chain(
        codecs[codec_name],
        chunk_mode == "encrypted",
        pack_mode == "encrypted",
    )
    lines = [f"stages\t{len(chain)}"]
    for index, fixture in enumerate(chain):
        manifest = fixture.manifest
        reference = manifest.states[0].chunks[0]
        prefix = f"stage.{index}"
        lines.extend(
            (
                f"{prefix}.manifest\t{kv.encode_manifest(manifest).hex()}",
                f"{prefix}.manifest_id\t{fixture.pack_id.hex()}",
                f"{prefix}.chunk\t{fixture.chunk.data.hex()}",
                f"{prefix}.chunk_id\t{reference.chunk_id.hex()}",
                f"{prefix}.object_key\t{reference.object_key.hex()}",
                f"{prefix}.object_digest\t{reference.object_digest.hex()}",
                f"{prefix}.pack\t{fixture.pack.hex()}",
                f"{prefix}.plaintext\t{fixture.plaintext.hex()}",
                f"{prefix}.token_start\t{fixture.span.token_start}",
                f"{prefix}.token_count\t{fixture.span.token_count}",
                f"{prefix}.plaintext_offset\t{fixture.span.plaintext_offset}",
                f"{prefix}.plaintext_bytes\t{fixture.span.plaintext_bytes}",
            )
        )
    return "\n".join(lines)


def verify_fixture(codec_name: str, pack_hex: str, chunk_hex: str) -> str:
    codecs = {"raw": kv.CODEC_RAW, "lossless": kv.CODEC_LOSSLESS}
    if codec_name not in codecs:
        raise SystemExit("unknown fixture codec")
    identity = lambda value: bytes((value,)) * 32
    keys = kv.derive_key_schedule(identity(99), identity(1), 7)
    manifest = kv.decode_pack(bytes.fromhex(pack_hex), keys)
    if manifest.family.states[0].codec != codecs[codec_name]:
        raise SystemExit("fixture codec mismatch")
    plaintext = kv.decode_chunk(
        bytes.fromhex(chunk_hex),
        manifest.states[0].chunks[0],
        manifest.realized_schema.states[0].chunk_spans[0],
        manifest.tenant_namespace,
        manifest.family,
        manifest.states[0].key,
        keys,
    )
    return plaintext.hex()


def verdict_chunk(codec_name: str, chunk_hex: str) -> str:
    codecs = {"raw": kv.CODEC_RAW, "lossless": kv.CODEC_LOSSLESS}
    if codec_name not in codecs:
        raise SystemExit("unknown fixture codec")
    fixture = kv.reference_fixture(codecs[codec_name], False, False)
    try:
        plaintext = kv.decode_chunk(
            bytes.fromhex(chunk_hex),
            fixture.manifest.states[0].chunks[0],
            fixture.span,
            fixture.manifest.tenant_namespace,
            fixture.manifest.family,
            fixture.manifest.states[0].key,
            fixture.keys,
        )
        return "ok\t" + plaintext.hex()
    except kv.WireError as error:
        return f"error\t{error.category}\t{error.message}"


def sidecar_fixture_lines() -> str:
    fixture = kv.reference_sidecar_fixture()
    family = fixture.family
    lines = [
        f"chunk\t{fixture.chunk.data.hex()}",
        f"sidecar\t{fixture.sidecar.hex()}",
        f"plaintext\t{fixture.plaintext.hex()}",
        f"chunk_id\t{fixture.reference.chunk_id.hex()}",
        f"object_key\t{fixture.reference.object_key.hex()}",
        f"object_digest\t{fixture.reference.object_digest.hex()}",
        f"family\t{kv.encode_family(family).hex()}",
        f"family_id\t{kv.representation_family_id(family).hex()}",
    ]
    return "\n".join(lines)


def verdict_sidecar_chunk(chunk_hex: str, object_digest_hex: str | None = None) -> str:
    fixture = kv.reference_sidecar_fixture()
    reference = fixture.reference
    if object_digest_hex is not None:
        reference = kv.ChunkRef(
            reference.chunk_id,
            reference.object_key,
            bytes.fromhex(object_digest_hex),
            reference.key_epoch,
            reference.plaintext_bytes,
            reference.object_bytes,
        )
    try:
        plaintext, sidecar = kv.decode_chunk_with_stats(
            bytes.fromhex(chunk_hex),
            reference,
            fixture.span,
            fixture.tenant,
            fixture.family,
            fixture.state_key,
            fixture.keys,
        )
        assert sidecar is not None
        return "ok\t" + plaintext.hex() + "\t" + kv.encode_sidecar(*sidecar).hex()
    except kv.WireError as error:
        return f"error\t{error.category}\t{error.message}"


def transcode(kind: str, encoded_hex: str) -> bytes:
    encoded = bytes.fromhex(encoded_hex)
    if kind == "state":
        return kv.encode_state_declaration(kv.decode_state_declaration(encoded))
    if kind == "family":
        return kv.encode_family(kv.decode_family(encoded, semantic=True))
    if kind == "schema":
        return kv.encode_schema(kv.decode_schema(encoded))
    if kind == "manifest":
        return kv.encode_manifest(kv.decode_manifest(encoded, semantic=True))
    if kind in {"raw-frame", "lossless-frame"}:
        if len(encoded) < 16:
            raise kv.WireError("truncated", "truncated codec frame")
        length = struct.unpack_from("<I", encoded, 12)[0]
        codec = kv.CODEC_RAW if kind == "raw-frame" else kv.CODEC_LOSSLESS
        return kv.encode_codec_frame(codec, kv.decode_codec_frame(codec, encoded, length))
    if kind == "pack":
        fixture = kv.reference_fixture(kv.CODEC_RAW, False, False)
        return kv.encode_manifest(kv.decode_pack(encoded, fixture.keys))
    raise SystemExit(f"unknown transcode kind {kind}")


def main(argv: list[str]) -> None:
    if argv == ["schema-vector"]:
        print(schema_vector().hex())
        return
    if len(argv) == 4 and argv[0] == "fixture-json":
        print(json.dumps(fixture_json(argv[1], argv[2], argv[3]), sort_keys=True))
        return
    if len(argv) == 4 and argv[0] == "fixture-lines":
        print(fixture_lines(argv[1], argv[2], argv[3]))
        return
    if len(argv) == 4 and argv[0] == "delta-chain-lines":
        print(delta_chain_lines(argv[1], argv[2], argv[3]))
        return
    if len(argv) == 4 and argv[0] == "verify-fixture":
        print(verify_fixture(argv[1], argv[2], argv[3]))
        return
    if argv == ["sidecar-fixture-lines"]:
        print(sidecar_fixture_lines())
        return
    if len(argv) in {2, 3} and argv[0] == "verdict-sidecar-chunk":
        print(verdict_sidecar_chunk(argv[1], argv[2] if len(argv) == 3 else None))
        return
    if len(argv) == 3 and argv[0] == "verdict-chunk":
        print(verdict_chunk(argv[1], argv[2]))
        return
    if len(argv) == 3 and argv[0] == "transcode":
        try:
            print("ok\t" + transcode(argv[1], argv[2]).hex())
        except kv.WireError as error:
            print(f"error\t{error.category}\t{error.message}")
        return
    raise SystemExit(
        "usage: production_v1.py schema-vector | fixture-json/fixture-lines/delta-chain-lines "
        "CODEC CHUNK PACK | sidecar-fixture-lines | "
        "verify-fixture CODEC PACK_HEX CHUNK_HEX | verdict-chunk CODEC CHUNK_HEX | "
        "verdict-sidecar-chunk CHUNK_HEX | "
        "transcode KIND HEX"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
