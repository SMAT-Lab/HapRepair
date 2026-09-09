# State And Reactivity Rules

Use this specification for state decorators, reactive ownership, component inputs,
and lifecycle mutations.

## Joint inspection

For each field, find its decorator, initializer, assignments, compound mutations,
collection mutations, UI reads, non-UI reads, parent arguments, child declarations,
watchers, `aboutToAppear`, `aboutToReuse`, and exported/public exposure. Analyze
`remove-redundant-state-var`, `remove-unchanged-state-var`, and
`use-local-var-to-replace-state-var` together when they overlap.

## Rules

- `hp-arkui-remove-unchanged-state-var`: remove reactive state only after proving the
  value never changes after initialization. Preserve a plain field when code still
  reads it; never delete a used declaration.
- `hp-arkui-remove-redundant-state-var`: determine whether the field is redundant,
  merely non-reactive, or derived from another source. Replace reads with the proven
  source or keep a plain field; preserve every non-UI use.
- `hp-arkui-use-local-var-to-replace-state-var`: compute a build-local value only
  when persistence across renders and external mutation are unnecessary. Preserve
  evaluation order and avoid recomputing expensive or side-effecting expressions.
- `multiple-associations-state-var-check`: split independent responsibilities or
  introduce a derived model without duplicating mutable ownership. Update all reads
  and writes consistently.
- `hp-arkui-no-state-var-access-in-loop`: snapshot a stable value before the loop
  only when the state cannot change during the loop. Do not move side effects or
  asynchronous reads across iteration boundaries.
- `hp-arkui-use-object-link-to-replace-prop`: change parent-child ownership only
  after checking that the value is an observed object supplied by a compatible
  parent. Migrate both sides and preserve initialization semantics.
- `hp-arkui-avoid-update-auto-state-var-in-aboutToReuse`: separate reusable input
  refresh from local state reset. Keep the original decorator unless the complete
  ownership analysis justifies a coordinated migration.
- `hp-arkui-no-func-as-arg-for-reusable-component`: replace captured callbacks with a
  stable event contract or method reference while preserving receiver binding,
  arguments, and lifecycle. Do not silently remove the callback.
- `hp-arkts-no-use-any-export-current` and `hp-arkts-no-use-any-export-other`: infer
  the narrowest compatible public type from all producers and consumers. Update the
  full import/export boundary; do not use casts that merely hide `any`.

## Interactions

State changes interact with reusable components, collection rendering, and lifecycle
callbacks. After editing a decorator or ownership boundary, inspect all
`@Reusable`, `aboutToReuse`, `@Link`, `@ObjectLink`, `@Prop`, and parent construction
sites on the same component before validation.

