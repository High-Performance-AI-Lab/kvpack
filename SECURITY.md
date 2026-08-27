# Security policy

Please report suspected vulnerabilities privately through GitHub's security
advisory interface. Do not open a public issue containing credentials,
exploitable inputs, private prompts, model data, or customer cache material.

The initial public candidate is pre-release software. Supported security fixes
target the current `0.1.0-alpha.*` line; older snapshots may receive no fixes.
Reports should include the exact commit, platform, minimal reproduction, and
the expected security boundary. Maintainers will acknowledge a report, assess
impact, coordinate remediation and disclosure, and credit reporters who wish
to be named.

kvpack validates and preserves engine-supplied state, but the integrating
engine remains responsible for synchronization, buffer meaning, and model
correctness. See `docs/ARCHITECTURE.md` for the trust boundary.
