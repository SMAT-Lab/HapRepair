# Collections, Rendering, And Reuse Rules

Use this specification for `ForEach`, `LazyForEach`, list/grid reuse, caching, and
preloading. Treat the render callback, key generator, item component, data source,
and surrounding container as one entity.

## Key and callback rules

- `foreach-args-check`: use callback parameters that match actual use. Preserve the
  render callback's index parameter when rendering depends on position.
- `foreach-index-check`: when the render callback accepts `(item, index)`, keep that
  callback shape and make the key generator accept the compatible parameters. Remove
  index from the key expression only when item identity itself is stable and unique.
  Otherwise use a repository-backed stable ID. Never recover an index with
  `collection.indexOf(item)`; duplicates are ambiguous and lookup is linear.
- `hp-arkui-no-stringify-in-lazyforeach-key-generator`: select an existing stable,
  primitive identity field or define an explicit stable ID at data creation. Do not
  replace serialization with a non-unique display value.

## Reuse and cache rules

- `hp-arkui-use-reusable-component`: extract or annotate a component only after
  verifying parameter compatibility, nested reusable restrictions, `aboutToReuse`
  refresh/reset logic, and stable reuse identity. Migrate all call sites together.
- `avoid-overusing-custom-component-check`: inline or replace a trivial component
  only when doing so preserves lifecycle, state isolation, styling scope, event
  handling, and reuse. Do not remove `@Reusable` mechanically to silence an
  interaction.
- `hp-arkui-suggest-reuseid-for-if-else-reusable-component`: assign reuse IDs that
  represent stable semantic variants; update all branches without conflating states.
- `hp-arkui-set-cache-count-for-lazyforeach-grid`: choose `cachedCount` from item
  cost, viewport size, memory pressure, and existing project conventions. Do not use
  a universal literal such as `2`. Attach the attribute to the API node required by
  the checker and SDK.
- `hp-arkui-use-grid-layout-options`: introduce layout options without changing item
  span, order, size, or responsive behavior.
- `init-list-component`: initialize the list component or controller according to
  its declared lifecycle; preserve current data and scroll position semantics.
- `hp-arkui-load-on-demand` and `waterflow-data-preload-check`: preserve data-source
  ordering, cache invalidation, end-of-list behavior, failure handling, and
  cancellation while moving work to demand/preload hooks.
- `hp-arkui-use-onAnimationStart-for-swiper-preload`: move preload initiation to
  `onAnimationStart` while preserving the target index, bounds, duplicate-request
  suppression, cancellation, and any state update previously performed by a later
  callback.

## Required joint validation

Whenever one of these rules is repaired, inspect all collection/reuse findings within
the same component. In particular, repair cache-count, Swiper preload, reusable
component, nested-component, and key-generator findings as one plan. Re-scan after
the joint edit and continue if a sibling alert appears.

