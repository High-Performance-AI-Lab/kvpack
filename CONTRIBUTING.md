# Contributing

Bug reports, documentation fixes, portability improvements, and focused tests
are welcome. Before submitting a change:

1. Keep generated output, models, dumps, credentials, personal paths, and lab
   routes out of the repository.
2. Preserve fail-closed validation and exact wire compatibility unless a
   versioned format change is explicitly approved.
3. Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --
   -D warnings`, `cargo test --workspace --all-targets --locked`,
   `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`, the wire-vector
   and LOC gates, manifest/dependency/notice checks, and `git diff --check`.
4. Add tests with behavior changes and document any platform-specific safety
   assumptions.

All contributions are accepted under both Apache-2.0 and MIT, matching the
repository license. Do not contribute code or data that you are not authorized
to redistribute.
