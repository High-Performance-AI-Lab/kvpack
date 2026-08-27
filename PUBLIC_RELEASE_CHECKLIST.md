# Public release checklist

- [x] Preserve `kvpack-handoff`'s exact revision while using public HTTPS.
- [x] Reject private dependency transports and non-exact Git revisions.
- [x] Regenerate the dependency notice inventory.
- [x] Replace personal model defaults with environment-based, skip-if-absent tests.
- [x] Require explicit lab host, key, binary, model, work and result locations.
- [x] Pass locked workspace tests, wire vectors, LOC, formatting and release clippy.
- [x] Pass conformance, tamper/fail-closed, C ABI and documentation examples.
- [x] Pass manifest, notice, dual-license and local draft-registry package gates.
- [ ] Export the deterministic clean candidate and pass its locked clean-clone gates.
- [x] Classify tree/history paths and blobs; select a fresh allowlisted snapshot.
- [ ] After `kvpack-cache` visibility approval, prove anonymous exact-revision fetch.
- [ ] Record candidate commit/tree, commands, toolchain and artifact hashes.

Visibility, package publication, tags, signing and GitHub releases are owner-only
actions and are not checked off by development work.
