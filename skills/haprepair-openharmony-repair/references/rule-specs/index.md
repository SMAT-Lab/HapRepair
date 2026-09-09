# Rule Specification Index

Always read `core-invariants.md`, then read every family specification returned by
the round plan. The family specification is normative. The 383-pair files under
`repair-guides/` are non-normative provenance examples and must not be copied as
patches.

| Family | Scope | Specification |
| --- | --- | --- |
| State and reactivity | Decorators, ownership, lifecycle state, public types | [state-reactivity.md](state-reactivity.md) |
| Collections and reuse | ForEach/LazyForEach, keys, reusable components, cache/preload | [collections-reuse.md](collections-reuse.md) |
| Layout, animation, media | View trees, transforms, visibility, presentation | [layout-animation-media.md](layout-animation-media.md) |
| Computation and API | Loops, allocation, callbacks, resources, concurrency | [computation-api.md](computation-api.md) |
| Structural security | Commented code and dependency cycles | [security-structure.md](security-structure.md) |
| Cryptographic security | Algorithms, keys, formats, storage, interoperability | [security-crypto.md](security-crypto.md) |

The manifest maps all 63 corpus rules plus the eight additional rules observed in
the frozen 35-project input. It also maps the two historical rule-ID spellings to the
current runtime identities.

