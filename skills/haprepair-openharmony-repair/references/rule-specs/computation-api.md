# Computation, Allocation, And API Rules

Use this specification for loop work, allocation, callbacks, platform APIs, and
concurrency.

## Local computation and allocation

- `constant-property-referencing-check-in-loops`: hoist only loop-invariant,
  side-effect-free expressions. Preserve getter calls, mutation visibility, and
  exception timing.
- `number-init-check`: preserve the numeric value and type. Do not replace a decimal
  literal with a convenient integer.
- `sparse-array-check`: replace sparse construction with a dense representation that
  preserves length, indices, holes-versus-undefined semantics, and consumers. Never
  replace the array with an unrelated sample value.
- `typed-array-check`: choose a typed array only when numeric range, signedness,
  precision, byte order, mutation, and API compatibility are proven.
- `reuse-date-instances-check` and `timezone-interface-check`: reuse objects or APIs
  without changing timezone, locale, daylight-saving, mutation, or formatting
  semantics.
- `hp-performance-no-closures`: replace a hot-path closure with a stable function or
  method while preserving captures, `this` binding, parameters, and lifetime.
- `hp-performance-no-dynamic-cls-func`: replace dynamic class/function creation with
  a statically declared equivalent while preserving lexical scope, constructor
  identity, prototypes, captures, and factory behavior.

## Calls and lifecycle

- `high-frequency-log-check`: reduce or guard logging without removing diagnostics
  required for errors, audit, or user support. Preserve argument side effects.
- `hp-arkui-avoid-empty-callback`: remove an optional empty callback only when the API
  does not use callback presence as a feature flag; otherwise implement the intended
  behavior.
- `js-code-cache-by-precompile-check`: update build/precompile configuration without
  changing runtime module loading or excluding required modules.
- `hp-arkui-suggest-cache-avplayer`: cache or reuse a player only with correct media
  source replacement, reset, lifecycle release, concurrency, and error handling.
- `hp-arkui-use-id-in-get-resource-sync-api`: replace name-based synchronous resource
  lookup with the generated resource ID corresponding to the same resource, variant,
  module, and type. Preserve fallback and exception behavior.
- `hp-arkui-use-taskpool-for-web-request`: move eligible request work to TaskPool while
  preserving network API compatibility, serialization constraints, UI-thread state
  updates, cancellation, errors, ordering, and lifecycle cleanup.

